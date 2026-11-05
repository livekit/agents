# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/llm/realtime.py

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from inspect import get_annotations
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeVar

from livekit import rtc

from ..llm.chat_context import ChatContext
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr, TranscriptChunk
from ..utils import is_given
from . import realtime_wire as wire
from .chat_items import MetricsMetadata
from .types import ProviderId

if TYPE_CHECKING:
    from ..llm.realtime import GenerationCreatedEvent
    from ..llm.tool_context import Tool, ToolChoice, ToolContext, Toolset


# Record declarations:
#     rust/crates/lk-agents-abi/src/llm/realtime.rs


@dataclass
class RealtimeCapabilities:
    message_truncation: bool
    """Whether generated assistant messages can be truncated after interruption"""
    turn_detection: bool
    """Whether the model emits server-side speech start and stop events for turn taking"""
    user_transcription: bool
    """Whether the model emits user audio transcription events"""
    auto_tool_reply_generation: bool
    """Whether the model automatically generates a reply after receiving tool results"""
    audio_output: bool
    """Whether the model can produce audio output directly"""
    manual_function_calls: bool
    """Whether function call items already in the chat context can be resumed"""
    can_disable_turn_detection: bool = False
    """Whether server-side turn detection can be disabled so the client drives turn-taking."""
    mutable_chat_context: bool = False
    """Whether the chat context can be updated mid-session"""
    mutable_instructions: bool = False
    """Whether the instructions can be updated mid-session"""
    mutable_tools: bool = False
    """Whether the tools can be updated mid-session"""
    per_response_tool_choice: bool = False
    """Whether the tool and tool choice can be specified per response"""
    supports_say: bool = False
    """Whether session.say() can use the realtime session directly, without TTS.

    When used through a RealtimeModel, add_to_chat_ctx=False is ignored and the
    message is still added to the chat context.
    """
    auto_turn_reply_generation: bool = False
    """Whether the server generates the reply by itself when its turn detection ends a user turn."""
    supports_overlapping_speech: bool = False
    """Whether the model and the caller may speak at once, the model deciding when to yield"""


class RealtimeModelError(BaseModel):
    """The public provider error event; its Python exception is retained but never serialized."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["realtime_model_error"] = "realtime_model_error"
    timestamp: float
    label: str
    error: Annotated[Exception, Field(exclude=True)]
    recoverable: bool


RealtimeCapabilities.__annotations__ = get_annotations(RealtimeCapabilities, eval_str=True)
RealtimeCapabilities.__module__ = "livekit.agents.llm.realtime"
RealtimeModelError.__module__ = "livekit.agents.llm"


class RealtimeModel(ABC):
    """A realtime model: speech in, speech out, over one live session.

    Plugins implement ``session()``, returning a ``RealtimeSession``.
    """

    def __init__(self, *, capabilities: RealtimeCapabilities) -> None:
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._capabilities = capabilities

    def _core_leg(self) -> object:
        """The Python provider implementation used by the native runtime's declared proxy."""
        return self

    @property
    def label(self) -> str:
        """The name logs and metrics give this instance: its class path."""
        return self._label

    @property
    def capabilities(self) -> RealtimeCapabilities:
        """What this instance supports, fixed at construction."""
        return self._capabilities

    @property
    def model(self) -> str:
        """The model name; ``"unknown"`` until a plugin overrides it."""
        return "unknown"

    @property
    def provider(self) -> str:
        """The provider name; ``"unknown"`` until a plugin overrides it."""
        return "unknown"

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """The model and provider labelling this instance's turn metrics."""
        return {"model_name": self.model, "model_provider": self.provider}

    @abstractmethod
    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        """Create a new session, optionally with server-side turn detection disabled.

        The model itself is left unchanged and reusable.

        Args:
            turn_detection_disabled: Hand turn-taking to the framework; honored only by plugins
                reporting ``can_disable_turn_detection``.
        """

    async def aclose(self) -> None:  # noqa: B027
        """Close the provider and everything it opened."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _core_instance_id(self) -> int | None:
        return id(self)

    def _core_label(self) -> str:
        return self.label

    def _core_capabilities(self) -> RealtimeCapabilities:
        return self.capabilities

    def _core_id(self) -> ProviderId:
        return {"label": self.label, "model": self.model, "provider": self.provider}

    async def _core_session(
        self, turn_detection_disabled: bool, tools: Callable[[], Sequence[Tool | Toolset]] = list
    ) -> RealtimeSession:
        out = self.session(turn_detection_disabled=turn_detection_disabled)
        out._core_attach(tools)
        return out

    def _core_with_tools(self, tools: Callable[[], Sequence[Tool | Toolset]]) -> Any:
        """The model as a session hands it to the framework: its sessions offer the tools
        ``tools`` names."""
        return _RealtimeModelTools(self, tools)


class _RealtimeModelTools:
    """``RealtimeModel`` as a session holds it: the tools its sessions offer."""

    def __init__(
        self, provider: RealtimeModel, tools: Callable[[], Sequence[Tool | Toolset]]
    ) -> None:
        self._provider = provider
        self._tools = tools

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    async def _core_session(self, turn_detection_disabled: bool) -> RealtimeSession:
        return await self._provider._core_session(turn_detection_disabled, self._tools)


EventTypes = Literal[
    "input_speech_started",  # serverside VAD (also used for interruptions)
    "input_speech_stopped",  # serverside VAD
    "input_audio_transcription_completed",
    "generation_created",
    "session_reconnected",
    "metrics_collected",
    "remote_item_added",
    "error",
]

TEvent = TypeVar("TEvent", default=Never, contravariant=True)


class RealtimeSession(ABC, rtc.EventEmitter[EventTypes | TEvent], Generic[TEvent]):
    """One live session of a realtime model.

    Plugins implement its commands (``push_audio``, ``generate_reply``, ``truncate``, the updates)
    and emit its events (``input_speech_started``, ``generation_created``, ``error``).
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__()
        self._realtime_model = realtime_model

    def _report_connection_acquired(self, acquire_time: float) -> None:
        """Report connection timing as a RealtimeModelMetrics event with zero usage."""
        from ..metrics import Metadata, RealtimeModelMetrics

        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id="",
                timestamp=time.time(),
                acquire_time=acquire_time,
                connection_reused=False,
                metadata=Metadata(
                    model_name=self._realtime_model.model,
                    model_provider=self._realtime_model.provider,
                ),
            ),
        )

    @property
    def realtime_model(self) -> RealtimeModel:
        return self._realtime_model

    @property
    def capabilities(self) -> RealtimeCapabilities:
        """Capabilities of the session.

        Defaults to the parent model's capabilities. Adapters that swap the underlying model
        mid-session override this to report the currently active model's capabilities.
        """
        return self._realtime_model.capabilities

    @property
    @abstractmethod
    def chat_ctx(self) -> ChatContext: ...

    @property
    @abstractmethod
    def tools(self) -> ToolContext: ...

    @abstractmethod
    async def update_instructions(self, instructions: str) -> None: ...

    @abstractmethod
    async def update_chat_ctx(
        self, chat_ctx: ChatContext
    ) -> None: ...  # can raise RealtimeError on Timeout

    @abstractmethod
    async def update_tools(self, tools: list[Tool]) -> None: ...

    @abstractmethod
    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None: ...

    @abstractmethod
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    @abstractmethod
    def push_video(self, frame: rtc.VideoFrame) -> None: ...

    @abstractmethod
    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]: ...  # can raise RealtimeError on Timeout

    # commit the input audio buffer to the server
    @abstractmethod
    def commit_audio(self) -> None: ...

    # clear the input audio buffer to the server
    @abstractmethod
    def clear_audio(self) -> None: ...

    # cancel the current generation (do nothing if no generation is in progress)
    @abstractmethod
    def interrupt(self) -> None: ...

    # message_id is the ID of the message to truncate (inside the ChatCtx)
    @abstractmethod
    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        from ..llm.realtime import RealtimeError

        if is_given(instructions):
            try:
                await self.update_instructions(instructions)
            except RealtimeError:
                logger.exception("failed to update the instructions")

        if is_given(chat_ctx):
            try:
                await self.update_chat_ctx(chat_ctx)
            except RealtimeError:
                logger.exception("failed to update the chat_ctx")

        if is_given(tools):
            try:
                await self.update_tools(tools)
            except RealtimeError:
                logger.exception("failed to update the tools")

    def start_user_activity(self) -> None:
        """notifies the model that user activity has started"""
        pass

    def say(
        self,
        text: str | AsyncIterable[str],
    ) -> asyncio.Future[GenerationCreatedEvent]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement say(). use a TTS model instead"
        )

    def _core_attach(self, tools: Callable[[], Sequence[Tool | Toolset]]) -> None:
        """Report this session's events to the framework, which offers it the tools ``tools``
        names."""
        from .._bindings import forward_session_events

        self._core_tools = tools
        self._core_events: asyncio.Queue[wire.Event | None] = asyncio.Queue()
        put = self._core_events.put_nowait
        forward_session_events(self, put)
        self.on(
            "generation_created",
            lambda ev: None if ev.user_initiated else put(wire.ServerGeneration(generation=ev)),
        )
        self.on(
            "remote_item_added",
            lambda ev: put(
                wire.RemoteItemAddedEvent(previous_item_id=ev.previous_item_id, item=ev.item)
            ),
        )

    def _core_send(self, cmd: wire.RealtimeCommand) -> None:
        match cmd:
            case wire.PushAudio(frame=frame):
                self.push_audio(frame)
            case wire.Interrupt():
                self.interrupt()
            case wire.CommitAudio():
                self.commit_audio()
            case wire.ClearAudio():
                self.clear_audio()
            case wire.Truncate():
                self.truncate(
                    message_id=cmd.message_id,
                    modalities=["audio", "text"] if cmd.has_audio else ["text"],
                    audio_end_ms=cmd.audio_end_ms,
                    audio_transcript=(
                        NOT_GIVEN if cmd.audio_transcript is None else cmd.audio_transcript
                    ),
                )
            case _:
                raise ValueError(f"unknown session command {cmd!r}")

    async def _core_update_instructions(self, instructions: str) -> None:
        await self.update_instructions(instructions)

    async def _core_update_chat_ctx(self, items: list[Any]) -> None:
        await self.update_chat_ctx(ChatContext(items))

    async def _core_push_item(self, item: Any) -> None:
        ctx = self.chat_ctx.copy()
        ctx.insert(item)
        await self.update_chat_ctx(ctx)

    async def _core_update_tools(self, names: list[str]) -> None:
        from .._bindings import offered_tools

        await self.update_tools(offered_tools(self._core_tools(), names))

    async def _core_request(self, cmd: wire.RealtimeCommand) -> asyncio.Future[Any]:
        match cmd:
            case wire.Say(text=text):
                return self.say(text)
            case wire.GenerateReply(instructions=instructions, tool_choice=choice):
                return self.generate_reply(
                    instructions=instructions or NOT_GIVEN,
                    tool_choice=NOT_GIVEN if choice is None else choice,
                )
        raise ValueError(f"unknown reply request {cmd!r}")

    async def _core_generation(
        self, request: asyncio.Future[GenerationCreatedEvent]
    ) -> GenerationCreatedEvent:
        return await request

    async def _core_cancel(self, request: asyncio.Future[Any]) -> None:
        request.cancel()

    async def _core_drain(self, ev: GenerationCreatedEvent, req_id: str | None) -> None:
        put = self._core_events.put_nowait
        response_id = ev.response_id or ""
        put(wire.GenerationCreated(response_id=response_id, req_id=req_id))

        async def _message(msg: Any) -> None:
            ids = {"message_id": msg.message_id, "response_id": ev.response_id}
            put(wire.MessageStarted(**ids))

            async def _text() -> None:
                async for delta in msg.text_stream:
                    put(
                        wire.TextDelta(delta=delta.text, start_time=delta.start_time, **ids)
                        if isinstance(delta, TranscriptChunk)
                        else wire.TextDelta(delta=delta, **ids)
                    )

            async def _audio() -> None:
                async for frame in msg.audio_stream:
                    put(wire.AudioDelta(frame=frame, **ids))

            async def _modalities() -> None:
                modalities = await msg.modalities
                kind: wire.Modalities = "audio" if "audio" in modalities else "text"
                put(wire.MessageModalities(modalities=kind, **ids))

            await asyncio.gather(_text(), _audio(), _modalities())
            put(wire.MessageDone(**ids))

        async def _messages() -> None:
            # each message forwards as it arrives, not once the server has sent them all
            started = []
            async for m in ev.message_stream:
                started.append(asyncio.ensure_future(_message(m)))
            await asyncio.gather(*started)

        async def _calls() -> None:
            async for call in ev.function_stream:
                put(wire.FunctionCallReady(call=call))

        await asyncio.gather(_messages(), _calls())
        put(wire.GenerationDone(response_id=response_id))

    async def _core_next_event(self) -> wire.Event | None:
        return await self._core_events.get()

    async def _core_aclose(self) -> None:
        await self.aclose()
        self._core_events.put_nowait(None)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()




# The public module identity is assigned after all classes are built.
RealtimeModel.__module__ = "livekit.agents.llm.realtime"
RealtimeSession.__module__ = "livekit.agents.llm.realtime"
