"""Translate native events into Python SDK events."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from livekit import rtc

from .. import _core, llm, stt, tts
from .._generated.amd import AMDPredictionEvent
from .._generated.session_dispatch import IGNORED_EVENTS, MIRRORED_STATES, PUBLIC_EVENTS
from .._generated.session_wire import (
    CloseWire,
    DeleteRoomRequested,
    HistoryReloadedWire,
    HostWire,
    ItemAddedWire,
    MetricsCollectedWire,
    OverlappingSpeechWire,
    RunItemAddedWire,
    RunRefusedWire,
    RunSettledWire,
    SessionErrorWire,
    SpeechCreatedWire,
    SpeechDoneWire,
    TextInputWire,
    ToolCallStartedWire,
    ToolExecutionUpdatedWire,
)
from ..job import get_job_context
from ..llm import (
    ChatContext,
)
from ..log import logger
from ._session_callbacks import (
    DefaultPipeline as DefaultPipeline,
    _AgentHooks as _AgentHooks,
)
from .agent import (
    _VoiceBinding,
)
from .agent_run import run_error
from .events import (
    AgentEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    ErrorEvent,
    EventTypes,
    SpeechCreatedEvent,
    ToolCallStarted,
    ToolExecutionUpdatedEvent,
    ToolUpdate,
    event_dict,
)
from .room_io import (
    AudioProcessingError,
    TextInputEvent,
)
from .run_result import _VoiceRun
from .speech_handle import SpeechHandle

if TYPE_CHECKING:
    from .agent_session import VoiceSession


def dispatch(self: VoiceSession[Any], wire: HostWire) -> None:
    kind = type(wire)
    if kind in IGNORED_EVENTS:
        return
    if (name := PUBLIC_EVENTS.get(kind)) is not None:
        if (attr := MIRRORED_STATES.get(kind)) is not None:
            setattr(self, attr, getattr(wire, "new_state"))  # noqa: B009
        self.emit(cast(EventTypes, name), cast(AgentEvent, wire))
        return
    handler = _HANDLERS.get(kind)
    if handler is not None:
        handler(self, wire)


def _on_speech_created(self: VoiceSession[Any], wire: SpeechCreatedWire) -> None:
    handle = SpeechHandle._of(cast(_core.SpeechHandle, wire.speech_handle), reported=False)
    self._speech_handles[handle.id] = handle
    self.emit(
        "speech_created",
        SpeechCreatedEvent(
            user_initiated=wire.user_initiated,
            source=cast(Literal["say", "generate_reply"], wire.source),
            speech_handle=handle,
        ),
    )


def _on_speech_done(self: VoiceSession[Any], wire: SpeechDoneWire) -> None:
    if (handle := self._speech_handles.pop(wire.speech_id, None)) is not None:
        handle._mark_done()


def _on_history_reloaded(self: VoiceSession[Any], wire: HistoryReloadedWire) -> None:
    self._chat_ctx = ChatContext(list(wire.items))


def _on_item_added(self: VoiceSession[Any], wire: ItemAddedWire) -> None:
    item = wire.item
    # an output told again changed in place (an answered question's)
    idx = self._chat_ctx.index_by_id(item.id) if item.type == "function_call_output" else None
    if idx is not None:
        self._chat_ctx.items[idx] = item
    else:
        self._chat_ctx.insert(item)
    if (run := _wire_run(wire.run)) is not None and item.type != "agent_handoff":
        run._item_added(item)
    if item.type in ("message", "agent_handoff"):
        self.emit("conversation_item_added", ConversationItemAddedEvent(item=item))


def _on_run_item(self: VoiceSession[Any], wire: RunItemAddedWire) -> None:
    if (run := _wire_run(wire.run)) is not None:
        run._item_added(wire.item)


def _on_run_refused(self: VoiceSession[Any], wire: RunRefusedWire) -> None:
    if (run := _wire_run(wire.run)) is not None:
        run._settle(output=None, error=RuntimeError(f"run refused: {wire.cause}"))


def _on_run_settled(self: VoiceSession[Any], wire: RunSettledWire) -> None:
    if (run := _wire_run(wire.run)) is None:
        return
    error: BaseException | None = None
    if (failure := wire.run_error) is not None:
        error = run_error(failure)
    speech = cast("_core.SpeechHandle | None", wire.speech_handle)
    if wire.outcome == "failed" and error is None and speech is not None:
        error = speech.exception()
    if wire.outcome == "closed":
        error = RuntimeError("the session closed before the run settled")
    run._settle(
        status=wire.outcome,
        output=wire.final_output,
        error=error,
        question=wire.question,
        input_requirements=list(wire.input_requirements),
    )


def _on_tool_update(self: VoiceSession[Any], wire: ToolExecutionUpdatedWire) -> None:
    update: ToolUpdate = (
        ToolCallStarted(function_call=wire.update.function_call)
        if isinstance(wire.update, ToolCallStartedWire)
        else wire.update
    )
    self.emit("tool_execution_updated", ToolExecutionUpdatedEvent(update=update))


def _on_metrics(self: VoiceSession[Any], wire: MetricsCollectedWire) -> None:
    # the usage counters are the core's, recorded where the session emits the record
    record = wire.metrics
    # a record the session metered for a provider (its VAD's) is that provider's event
    if (leg := wire.leg) is not None and (
        provider := getattr(_VoiceBinding(self, self.current_agent), leg)
    ) is not None:
        provider.emit("metrics_collected", record)


def _error_record(self: VoiceSession[Any], wire: SessionErrorWire) -> tuple[Any, Any]:
    """The reference's payload: the provider's error record, and the provider that failed."""
    if (failure := wire.processor_failure) is not None:
        error = AudioProcessingError(label=failure.label, error=cast(Exception, failure.error))
        return error, failure.processor
    record, leg = _ERROR_RECORDS[wire.source]
    source = getattr(_VoiceBinding(self, self.current_agent), leg)
    error = record(
        timestamp=time.time(),
        label=getattr(source, "label", wire.label),
        error=wire.error,
        recoverable=wire.recoverable,
    )
    return error, source


def _core_error_message(self: VoiceSession[Any], wire: SessionErrorWire) -> str:
    return str(_error_record(self, wire)[0])


def _on_error(self: VoiceSession[Any], wire: SessionErrorWire) -> None:
    # A recoverable error is one the call goes on through; a processor's own exception keeps
    # its traceback.
    failure = wire.processor_failure
    logger.log(
        logging.WARNING if wire.recoverable else logging.ERROR,
        "session error",
        extra={"source": wire.source, "error": str(wire.error)},
        exc_info=cast(BaseException, failure.error) if failure is not None else None,
    )
    error, source = _error_record(self, wire)
    self.emit("error", ErrorEvent(error=error, source=source))


def _on_overlapping_speech(self: VoiceSession[Any], wire: OverlappingSpeechWire) -> None:
    self.emit("overlapping_speech", wire.event)


def _on_amd_result(self: VoiceSession[Any], wire: AMDPredictionEvent) -> None:
    if self._amd is not None:
        self._amd._on_prediction(wire)


def _on_text_input(self: VoiceSession[Any], wire: TextInputWire) -> None:
    if (callback := self._text_input_cb) is None:
        return
    room = self._room
    participants = room.remote_participants if isinstance(room, rtc.Room) else {}
    ev = TextInputEvent(
        text=wire.text,
        info=wire.info if isinstance(wire.info, rtc.TextStreamInfo) else None,
        participant=participants.get(wire.participant),
    )

    async def _run() -> None:
        try:
            if inspect.isawaitable(result := callback(self, ev)):
                await result
        except Exception:
            logger.warning("failed to handle chat text stream", exc_info=True)

    task = asyncio.ensure_future(_run())
    self._tasks.add(task)
    task.add_done_callback(self._tasks.discard)


def _on_delete_room(self: VoiceSession[Any], wire: DeleteRoomRequested) -> None:
    if (job_ctx := get_job_context(required=False)) is None:
        return
    logger.info(
        "deleting room on agent session close "
        "(disable via `RoomOptions.delete_room_on_close=False`)",
        extra={"room": wire.room},
    )
    job_ctx.delete_room(room_name=wire.room)


def _on_close(self: VoiceSession[Any], wire: CloseWire) -> None:
    error = _error_record(self, cause)[0] if (cause := wire.error) is not None else None
    self.emit("close", CloseEvent(reason=CloseReason(wire.reason), error=error))


_ERROR_RECORDS: dict[str, tuple[Any, str]] = {
    "llm_error": (llm.LLMError, "llm"),
    "tts_error": (tts.TTSError, "tts"),
    "stt_error": (stt.STTError, "stt"),
    "realtime_model_error": (llm.RealtimeModelError, "llm"),
}


def _wire_run(run: object | None) -> _VoiceRun | None:
    """A wire's run: the `_VoiceRun` the host handed the core."""
    return cast("_VoiceRun | None", run)


# The `HostWire` members with host behaviour, by record class.
_HANDLERS: dict[type, Callable[[VoiceSession[Any], Any], None]] = {
    HistoryReloadedWire: _on_history_reloaded,
    SpeechCreatedWire: _on_speech_created,
    SpeechDoneWire: _on_speech_done,
    ItemAddedWire: _on_item_added,
    RunItemAddedWire: _on_run_item,
    RunRefusedWire: _on_run_refused,
    RunSettledWire: _on_run_settled,
    ToolExecutionUpdatedWire: _on_tool_update,
    MetricsCollectedWire: _on_metrics,
    OverlappingSpeechWire: _on_overlapping_speech,
    SessionErrorWire: _on_error,
    CloseWire: _on_close,
    AMDPredictionEvent: _on_amd_result,
    TextInputWire: _on_text_input,
    DeleteRoomRequested: _on_delete_room,
}


def report_snapshots(self: VoiceSession[Any]) -> list[dict[str, Any]]:
    expired = self._event_retention.expire(time.monotonic_ns())
    for _ in range(expired):
        self._event_snapshots.popleft()
    return [
        event
        for event in self._event_snapshots
        if self._event_retention.accepts(event["type"], self._recording_options["redaction"])
    ]


def record_snapshot(self: VoiceSession[Any], arg: AgentEvent) -> None:
    # the event as the report records it: no live objects (a model client, an exception)
    if self._event_retention.accepts(arg.type, self._recording_options["redaction"]):
        snapshot = event_dict(arg, json=True)
        expired, admitted = self._event_retention.admit(snapshot, time.monotonic_ns())
        for _ in range(expired):
            self._event_snapshots.popleft()
        if admitted:
            self._event_snapshots.append(snapshot)
