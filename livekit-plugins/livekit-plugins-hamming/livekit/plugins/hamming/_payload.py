from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from livekit.agents.voice.events import CloseEvent

from .log import logger

_CALL_ID_METADATA_KEY = "call_id"
CALL_ID_STRATEGY_ROOM_NAME = "room_name"
CALL_ID_STRATEGY_PARTICIPANT_IDENTITY = "participant_identity"
CALL_ID_STRATEGY_PARTICIPANT_METADATA = "participant_metadata"
CALL_ID_STRATEGY_CUSTOM = "custom"


class SessionReportLike(Protocol):
    @property
    def room(self) -> str: ...

    @property
    def started_at(self) -> float | None: ...

    @property
    def timestamp(self) -> float: ...

    @property
    def events(self) -> list[Any]: ...

    @property
    def audio_recording_path(self) -> Any | None: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CallIdResolutionContext:
    room_name: str
    participant_identity: str | None
    participant_metadata_raw: str | None
    external_agent_id: str


CallIdResolver = Callable[[CallIdResolutionContext], str | None]


@dataclass(frozen=True)
class PayloadBuildConfig:
    external_agent_id: str
    plugin_api_version: str
    plugin_version: str
    payload_schema_version: str
    call_id_strategy: str = CALL_ID_STRATEGY_ROOM_NAME
    call_id_metadata_key: str = _CALL_ID_METADATA_KEY
    resolve_call_id: CallIdResolver | None = None
    capture_manifest: Mapping[str, Any] | None = None


def build_livekit_monitoring_envelope(
    *,
    config: PayloadBuildConfig,
    report: SessionReportLike,
    participant_identity: str | None,
    participant_metadata_raw: str | None,
    recording_context: Mapping[str, Any] | None,
    close_event: CloseEvent | None,
) -> dict[str, Any]:
    call_id = _resolve_call_id(
        room_name=report.room,
        participant_identity=participant_identity,
        participant_metadata_raw=participant_metadata_raw,
        external_agent_id=config.external_agent_id,
        strategy=config.call_id_strategy,
        metadata_key=config.call_id_metadata_key,
        resolve_call_id=config.resolve_call_id,
    )
    test_case_run_id = _resolve_test_case_run_id(
        participant_metadata_raw=participant_metadata_raw,
        recording_context=recording_context,
    )

    payload = {
        "call_id": call_id,
        "call_type": "web",
        "livekit_room_name": report.room,
        "start_timestamp": int((report.started_at or report.timestamp) * 1000),
        "end_timestamp": int(report.timestamp * 1000),
        "status": _resolve_status(close_event),
        "livekit_capture": _build_livekit_capture(
            report=report,
            participant_identity=participant_identity,
            participant_metadata_raw=participant_metadata_raw,
            close_event=close_event,
        ),
    }
    if test_case_run_id:
        payload["test_case_run_id"] = test_case_run_id

    metadata: dict[str, Any] = {
        "integration": "livekit-plugin-hamming",
        "mode": "call_review",
        "call_id_strategy": config.call_id_strategy,
    }
    if config.capture_manifest:
        metadata["capture_manifest"] = dict(config.capture_manifest)

    return {
        "provider": "custom",
        "external_agent_id": config.external_agent_id,
        "payload_schema_version": config.payload_schema_version,
        "plugin_api_version": config.plugin_api_version,
        "plugin_version": config.plugin_version,
        "payload": payload,
        "metadata": metadata,
    }


def _build_livekit_capture(
    *,
    report: SessionReportLike,
    participant_identity: str | None,
    participant_metadata_raw: str | None,
    close_event: CloseEvent | None,
) -> dict[str, Any]:
    capture = report.to_dict()
    capture["started_at"] = report.started_at
    capture["timestamp"] = report.timestamp
    capture["participant_identity"] = participant_identity
    if participant_metadata_raw:
        capture["participant_metadata"] = participant_metadata_raw
    capture["close_reason"] = _serialize_close_reason(close_event)
    capture["events"] = _serialize_events(report.events)
    return capture


def _serialize_events(events: list[Any]) -> list[dict[str, Any]]:
    events_out: list[dict[str, Any]] = []

    for event in events:
        try:
            raw = event.model_dump(mode="json", exclude_none=False)
        except Exception:
            logger.debug(
                "failed to serialize session event for livekit capture",
                extra={"event_type": type(event).__name__},
                exc_info=True,
            )
            continue

        if isinstance(raw, dict) and isinstance(raw.get("type"), str):
            events_out.append(raw)

    return events_out


def _resolve_call_id(
    *,
    room_name: str,
    participant_identity: str | None,
    participant_metadata_raw: str | None,
    external_agent_id: str,
    strategy: str = CALL_ID_STRATEGY_ROOM_NAME,
    metadata_key: str = _CALL_ID_METADATA_KEY,
    resolve_call_id: CallIdResolver | None = None,
) -> str:
    fallback = room_name

    if strategy == CALL_ID_STRATEGY_ROOM_NAME:
        return fallback

    if strategy == CALL_ID_STRATEGY_PARTICIPANT_IDENTITY:
        return _resolved_string_or_fallback(participant_identity, fallback)

    if strategy == CALL_ID_STRATEGY_PARTICIPANT_METADATA:
        metadata = _parse_metadata(participant_metadata_raw)
        return _resolved_string_or_fallback(metadata.get(metadata_key), fallback)

    if strategy == CALL_ID_STRATEGY_CUSTOM and resolve_call_id is not None:
        return _resolve_custom_call_id(
            room_name=room_name,
            participant_identity=participant_identity,
            participant_metadata_raw=participant_metadata_raw,
            external_agent_id=external_agent_id,
            resolve_call_id=resolve_call_id,
            fallback=fallback,
        )

    return fallback


def _resolve_custom_call_id(
    *,
    room_name: str,
    participant_identity: str | None,
    participant_metadata_raw: str | None,
    external_agent_id: str,
    resolve_call_id: CallIdResolver,
    fallback: str,
) -> str:
    try:
        resolved_call_id = resolve_call_id(
            CallIdResolutionContext(
                room_name=room_name,
                participant_identity=participant_identity,
                participant_metadata_raw=participant_metadata_raw,
                external_agent_id=external_agent_id,
            )
        )
    except Exception:
        logger.warning(
            "custom call_id resolver failed; falling back to room name",
            extra={"room_name": room_name},
            exc_info=True,
        )
        return fallback

    return _resolved_string_or_fallback(resolved_call_id, fallback)


def _resolved_string_or_fallback(value: object, fallback: str) -> str:
    if value is None:
        return fallback

    candidate = str(value).strip()
    return candidate or fallback


def _parse_metadata(metadata_raw: str | None) -> dict[str, Any]:
    if not metadata_raw:
        return {}

    try:
        parsed = json.loads(metadata_raw)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return parsed

    return {}


def _resolve_test_case_run_id(
    *,
    participant_metadata_raw: str | None,
    recording_context: Mapping[str, Any] | None,
) -> str | None:
    metadata = _parse_metadata(participant_metadata_raw)
    for key in (
        "test_case_run_id",
        "testCaseRunId",
        "conversation_id",
        "conversationId",
    ):
        value = metadata.get(key)
        if value is None:
            continue
        candidate = str(value).strip()
        if candidate:
            return candidate

    if not recording_context:
        return None

    for key in (
        "customer_conversation_id",
        "test_case_run_id",
        "testCaseRunId",
        "conversation_id",
        "conversationId",
    ):
        value = recording_context.get(key)
        if value is None:
            continue
        candidate = str(value).strip()
        if candidate:
            return candidate

    return None


def _resolve_status(close_event: CloseEvent | None) -> str:
    return "error" if _serialize_close_reason(close_event) == "error" else "ended"


def _serialize_close_reason(close_event: CloseEvent | None) -> str | None:
    if close_event is None:
        return None

    reason = close_event.reason
    if reason is None:
        return None
    if isinstance(reason, str):
        return reason

    enum_value = getattr(reason, "value", None)
    if isinstance(enum_value, str):
        return enum_value

    return str(reason)
