from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any, TypedDict, cast

from livekit.agents.job import get_job_context
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.events import CloseEvent

from ._payload import (
    CALL_ID_STRATEGY_CUSTOM,
    CALL_ID_STRATEGY_PARTICIPANT_IDENTITY,
    CALL_ID_STRATEGY_PARTICIPANT_METADATA,
    CALL_ID_STRATEGY_ROOM_NAME,
    CallIdResolver,
    PayloadBuildConfig,
    SessionReportLike,
    build_livekit_monitoring_envelope,
)
from ._transport import ConnectionPolicy, HammingTransport
from .log import logger
from .version import __version__

if TYPE_CHECKING:
    from livekit.agents.job import JobContext


RECORDING_MODE_NONE = "none"
RECORDING_MODE_SESSION_AUDIO = "session_audio"
RECORDING_MODE_ROOM_COMPOSITE = "room_composite"
RECORDING_MODE_PARTICIPANT_EGRESS = "participant_egress"
SUPPORTED_RECORDING_MODES = {
    RECORDING_MODE_NONE,
    RECORDING_MODE_SESSION_AUDIO,
    RECORDING_MODE_ROOM_COMPOSITE,
    RECORDING_MODE_PARTICIPANT_EGRESS,
}

RECORDING_SOURCE_EXTERNAL_URLS = "external_urls"
RECORDING_SOURCE_AUTO_EGRESS = "auto_egress"
LEGACY_RECORDING_SOURCE_CUSTOMER_MANAGED = "customer_managed"
LEGACY_RECORDING_SOURCE_PLUGIN_MANAGED = "plugin_managed"
SUPPORTED_RECORDING_SOURCES = {
    RECORDING_SOURCE_EXTERNAL_URLS,
    RECORDING_SOURCE_AUTO_EGRESS,
}

RECORDING_DELIVERY_AUTO = "auto"
RECORDING_DELIVERY_URL = "url"
RECORDING_DELIVERY_UPLOAD = "upload"
SUPPORTED_RECORDING_DELIVERIES = {
    RECORDING_DELIVERY_AUTO,
    RECORDING_DELIVERY_URL,
    RECORDING_DELIVERY_UPLOAD,
}

SUPPORTED_CALL_ID_STRATEGIES = {
    CALL_ID_STRATEGY_ROOM_NAME,
    CALL_ID_STRATEGY_PARTICIPANT_IDENTITY,
    CALL_ID_STRATEGY_PARTICIPANT_METADATA,
    CALL_ID_STRATEGY_CUSTOM,
}
SUPPORTED_CAPTURE_MODES = {"none", "full"}
SAMPLING_VERSION = "v1"

STREAMING_MODE_NONE = "none"
STREAMING_MODE_TRACK_EGRESS = "track_egress"
SUPPORTED_STREAMING_MODES = {
    STREAMING_MODE_NONE,
    STREAMING_MODE_TRACK_EGRESS,
}

TEST_CASE_RUN_ID_METADATA_KEYS = (
    "test_case_run_id",
    "testCaseRunId",
    "conversation_id",
    "conversationId",
)
PLUGIN_MANAGED_RECORDINGS_DIR = "recordings"
PLUGIN_MANAGED_ROOM_COMPOSITE_FILENAME_PREFIX = "hamming_plugin_room_composite"
PLUGIN_MANAGED_ROOM_COMPOSITE_EXTENSION = "ogg"


class RecordingContext(TypedDict, total=False):
    call_id: str
    room_name: str
    customer_conversation_id: str
    provider_recording_id: str
    provider_recording_filepath: str
    metadata: dict[str, str]


class RecordingArtifacts(TypedDict, total=False):
    recording_url: str
    agent_recording_url: str
    user_recording_url: str
    local_paths: list[str]


class RecordingConfigInput(TypedDict, total=False):
    mode: str
    source: str
    delivery: str
    resolver: RecordingResolver


class StreamingConfigInput(TypedDict, total=False):
    mode: str
    source: str


class CaptureConfigInput(TypedDict, total=False):
    agent_context: bool
    prompts: str
    tools: str
    interim_transcripts: bool


class RedactConfigInput(TypedDict, total=False):
    pii: bool
    tool_args_paths: list[str]
    tool_output_paths: list[str]


class LiveKitConfigInput(TypedDict, total=False):
    url: str
    api_key: str
    api_secret: str


@dataclass(frozen=True)
class RecordingResolutionContext:
    session: AgentSession[Any]
    report: SessionReportLike
    job_ctx: JobContext | None
    close_event: CloseEvent | None
    call_id: str
    room_name: str
    participant_identity: str | None
    participant_metadata_raw: str | None
    external_agent_id: str
    recording_context: RecordingContext | None


RecordingResolverResult = RecordingArtifacts | Awaitable[RecordingArtifacts]
RecordingResolver = Callable[[RecordingResolutionContext], RecordingResolverResult]


@dataclass(frozen=True)
class ResolvedRecordingConfig:
    mode: str = RECORDING_MODE_NONE
    source: str = RECORDING_SOURCE_AUTO_EGRESS
    delivery: str = RECORDING_DELIVERY_AUTO
    resolver: RecordingResolver | None = None


@dataclass(frozen=True)
class ResolvedStreamingConfig:
    mode: str = STREAMING_MODE_NONE
    source: str = RECORDING_SOURCE_EXTERNAL_URLS


@dataclass(frozen=True)
class ResolvedCaptureConfig:
    agent_context: bool = True
    prompts: str = "full"
    tools: str = "full"
    interim_transcripts: bool = False


@dataclass(frozen=True)
class ResolvedRedactConfig:
    pii: bool = True
    tool_args_paths: tuple[str, ...] = ()
    tool_output_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedLiveKitConfig:
    url: str | None = None
    api_key: str | None = None
    api_secret: str | None = None


@dataclass(frozen=True)
class ResolvedSamplingConfig:
    rate: float = 1.0
    key: str = "participant_identity"


@dataclass(frozen=True)
class PluginManagedRoomCompositeState:
    room_name: str
    filepath: str
    start_task: asyncio.Task[str]


@dataclass(frozen=True)
class HammingConfig:
    api_key: str
    base_url: str = "https://app.hamming.ai"
    call_review_enabled: bool = True
    flush_on_shutdown: bool = True
    auto_record_audio: bool = False
    external_agent_id: str | None = None
    plugin_api_version: str = "1.0.0"
    request_timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5
    payload_schema_version: str = "2026-03-02"
    call_id_strategy: str = CALL_ID_STRATEGY_ROOM_NAME
    call_id_metadata_key: str = "call_id"
    resolve_call_id: CallIdResolver | None = None
    sampling: ResolvedSamplingConfig = ResolvedSamplingConfig()
    capture: ResolvedCaptureConfig = ResolvedCaptureConfig()
    redact: ResolvedRedactConfig = ResolvedRedactConfig()
    livekit: ResolvedLiveKitConfig = ResolvedLiveKitConfig()
    recording: ResolvedRecordingConfig = ResolvedRecordingConfig()
    streaming: ResolvedStreamingConfig = ResolvedStreamingConfig()


class _SessionMonitor:
    def __init__(
        self,
        *,
        runtime: HammingRuntime,
        session: AgentSession[Any],
        participant_identity: str | None,
        participant_metadata: str | None,
        external_agent_id: str,
        job_ctx: JobContext | None,
        session_key: int,
        recording_context: RecordingContext | None,
    ) -> None:
        self._runtime = runtime
        self._session = session
        self._participant_identity = participant_identity
        self._participant_metadata = participant_metadata
        self._external_agent_id = external_agent_id
        self._job_ctx = job_ctx
        self._session_key = session_key
        self._recording_context = (
            cast(RecordingContext, dict(recording_context)) if recording_context else None
        )
        self._plugin_managed_room_composite: PluginManagedRoomCompositeState | None = None
        self._close_event: CloseEvent | None = None
        self._close_event_ready = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._send_task: asyncio.Task[None] | None = None

    def on_close(self, event: CloseEvent) -> None:
        self._close_event = event
        self._close_event_ready.set()
        self._ensure_send_task(from_shutdown=False)

    async def on_shutdown(self, reason: str | None = None) -> None:
        if not self._runtime.config.flush_on_shutdown:
            return

        task = self._ensure_send_task(from_shutdown=True)
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("timed out waiting for hamming final payload flush")
        except asyncio.CancelledError:
            logger.warning(
                "hamming shutdown callback cancelled before flush completed",
                extra={"reason": reason},
            )
        except Exception:
            logger.exception(
                "hamming shutdown callback failed while waiting for flush",
                extra={"reason": reason},
            )

    def start_plugin_managed_room_composite_if_needed(self) -> None:
        recording = self._runtime.config.recording
        if (
            recording.mode != RECORDING_MODE_ROOM_COMPOSITE
            or recording.source != RECORDING_SOURCE_AUTO_EGRESS
        ):
            return

        if self._job_ctx is None:
            logger.warning(
                "plugin-managed room composite requires JobContext; falling back to resolver path",
            )
            return

        filepath = _build_plugin_managed_room_composite_filepath(
            room_name=self._job_ctx.room.name,
            participant_metadata_raw=self._participant_metadata,
            recording_context=self._recording_context,
            session_key=self._session_key,
        )
        logger.info(
            "scheduling plugin-managed room composite egress",
            extra={
                "room_name": self._job_ctx.room.name,
                "filepath": filepath,
                **_recording_debug_context(
                    participant_metadata_raw=self._participant_metadata,
                    recording_context=self._recording_context,
                ),
            },
        )
        start_task = asyncio.create_task(
            self._start_plugin_managed_room_composite_egress(filepath=filepath)
        )
        self._runtime.track_task(start_task)
        self._plugin_managed_room_composite = PluginManagedRoomCompositeState(
            room_name=self._job_ctx.room.name,
            filepath=filepath,
            start_task=start_task,
        )

    def _ensure_send_task(self, *, from_shutdown: bool) -> asyncio.Task[None]:
        if self._send_task is None:
            self._send_task = asyncio.create_task(
                self._send_final_payload(wait_for_close_event=from_shutdown)
            )
            self._runtime.track_task(self._send_task)

        return self._send_task

    async def _send_final_payload(self, *, wait_for_close_event: bool) -> None:
        try:
            async with self._send_lock:
                close_event = await self._wait_for_close_event_if_needed(
                    wait_for_close_event=wait_for_close_event
                )
                report = self._build_report_for_export()
                if report is None:
                    return

                envelope = self._build_monitoring_envelope(report=report, close_event=close_event)
                payload_record = envelope.get("payload")
                if not isinstance(payload_record, dict):
                    await self._send_envelope(envelope=envelope, report=report, payload_record=None)
                    return

                if self._should_skip_for_sampling(
                    envelope=envelope,
                    payload_record=payload_record,
                    report=report,
                ):
                    return

                await self._populate_recording_payload(
                    report=report,
                    payload_record=payload_record,
                    close_event=close_event,
                )
                self._log_recording_resolution_outcome(
                    report=report,
                    payload_record=payload_record,
                )
                await self._send_envelope(
                    envelope=envelope,
                    report=report,
                    payload_record=payload_record,
                )
        except Exception as exc:
            logger.error("unhandled exception in hamming final payload task: %r", exc)
            logger.exception("unhandled exception in hamming final payload task")
        finally:
            self._runtime.remove_monitor(self._session_key)

    async def _wait_for_close_event_if_needed(
        self,
        *,
        wait_for_close_event: bool,
    ) -> CloseEvent | None:
        if wait_for_close_event and self._close_event is None:
            try:
                await asyncio.wait_for(self._close_event_ready.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

        return self._close_event

    def _build_report_for_export(self) -> SessionReportLike | None:
        if not self._runtime.config.call_review_enabled:
            logger.debug("hamming call review export disabled; skipping final payload")
            return None

        try:
            return self._runtime.build_report(self._session, self._job_ctx)
        except Exception:
            logger.exception("failed to build session report for hamming payload")
            return None

    def _build_monitoring_envelope(
        self,
        *,
        report: SessionReportLike,
        close_event: CloseEvent | None,
    ) -> dict[str, Any]:
        envelope = build_livekit_monitoring_envelope(
            config=PayloadBuildConfig(
                external_agent_id=self._external_agent_id,
                plugin_api_version=self._runtime.config.plugin_api_version,
                plugin_version=__version__,
                payload_schema_version=self._runtime.config.payload_schema_version,
                call_id_strategy=self._runtime.config.call_id_strategy,
                call_id_metadata_key=self._runtime.config.call_id_metadata_key,
                resolve_call_id=self._runtime.config.resolve_call_id,
                capture_manifest=_build_capture_manifest(self._runtime.config),
            ),
            report=report,
            participant_identity=self._participant_identity,
            participant_metadata_raw=self._participant_metadata,
            recording_context=self._recording_context,
            close_event=close_event,
        )

        enable_test_case_matching_override = _env_optional_bool("HAMMING_ENABLE_TEST_CASE_MATCHING")
        if enable_test_case_matching_override is not None:
            envelope["enableTestCaseMatching"] = enable_test_case_matching_override
        return envelope

    def _should_skip_for_sampling(
        self,
        *,
        envelope: dict[str, Any],
        payload_record: dict[str, Any],
        report: SessionReportLike,
    ) -> bool:
        sampling_metadata = _build_sampling_metadata(
            config=self._runtime.config,
            call_id=_string_or_fallback(payload_record.get("call_id"), report.room),
            room_name=report.room,
            participant_identity=self._participant_identity,
            participant_metadata_raw=self._participant_metadata,
            external_agent_id=self._external_agent_id,
        )
        metadata_record = envelope.setdefault("metadata", {})
        if isinstance(metadata_record, dict):
            metadata_record["sampling"] = sampling_metadata
        if sampling_metadata["decision"] != "excluded":
            return False

        logger.info(
            "skipping hamming payload due to deterministic sampling",
            extra={
                "call_id": payload_record.get("call_id"),
                "room_name": report.room,
                "sampling_key": sampling_metadata["key"],
                "sampling_rate": sampling_metadata["rate"],
            },
        )
        return True

    async def _send_envelope(
        self,
        *,
        envelope: dict[str, Any],
        report: SessionReportLike,
        payload_record: dict[str, Any] | None,
    ) -> None:
        try:
            await self._runtime.transport.send_capture(envelope)
        except Exception:
            logger.exception("failed to send hamming payload")
            return

        logger.info(
            "sent hamming payload",
            extra={
                "call_id": payload_record.get("call_id") if payload_record is not None else None,
                "room_name": report.room,
            },
        )

    async def _populate_recording_payload(
        self,
        *,
        report: SessionReportLike,
        payload_record: dict[str, Any],
        close_event: CloseEvent | None,
    ) -> None:
        recording = self._runtime.config.recording
        if recording.mode == RECORDING_MODE_NONE:
            return

        if recording.mode == RECORDING_MODE_SESSION_AUDIO:
            await self._attach_inline_recording_capture(
                report=report, payload_record=payload_record
            )
            return

        if recording.mode not in {
            RECORDING_MODE_ROOM_COMPOSITE,
            RECORDING_MODE_PARTICIPANT_EGRESS,
        }:
            logger.warning("unsupported hamming recording mode", extra={"mode": recording.mode})
            return

        resolution_context = self._build_recording_resolution_context(
            report=report,
            payload_record=payload_record,
            close_event=close_event,
        )
        if await self._populate_builtin_recording_payload(
            payload_record=payload_record,
            resolution_context=resolution_context,
        ):
            return

        resolved_artifacts = await self._resolve_recording_artifacts(
            payload_record=payload_record,
            resolution_context=resolution_context,
        )
        if resolved_artifacts is None:
            return

        self._apply_resolved_recording_artifacts(
            payload_record=payload_record,
            resolved_artifacts=resolved_artifacts,
        )

    def _build_recording_resolution_context(
        self,
        *,
        report: SessionReportLike,
        payload_record: dict[str, Any],
        close_event: CloseEvent | None,
    ) -> RecordingResolutionContext:
        return RecordingResolutionContext(
            session=self._session,
            report=report,
            job_ctx=self._job_ctx,
            close_event=close_event,
            call_id=_string_or_fallback(payload_record.get("call_id"), report.room),
            room_name=report.room,
            participant_identity=self._participant_identity,
            participant_metadata_raw=self._participant_metadata,
            external_agent_id=self._external_agent_id,
            recording_context=self._recording_context,
        )

    async def _populate_builtin_recording_payload(
        self,
        *,
        payload_record: dict[str, Any],
        resolution_context: RecordingResolutionContext,
    ) -> bool:
        recording = self._runtime.config.recording
        if recording.mode != RECORDING_MODE_ROOM_COMPOSITE:
            return False

        if recording.source == RECORDING_SOURCE_AUTO_EGRESS:
            recording_url = await self._resolve_plugin_managed_room_composite_recording_url(
                resolution_context=resolution_context,
            )
            if recording_url:
                payload_record["recording_url"] = recording_url
            return True

        recording_url = await self._resolve_hamming_room_composite_recording_url(
            resolution_context=resolution_context,
        )
        if recording_url:
            payload_record["recording_url"] = recording_url
            return True
        return False

    async def _resolve_recording_artifacts(
        self,
        *,
        payload_record: dict[str, Any],
        resolution_context: RecordingResolutionContext,
    ) -> RecordingArtifacts | None:
        recording = self._runtime.config.recording
        resolver = recording.resolver
        if resolver is None:
            logger.warning(
                "recording mode requires resolver but none was configured",
                extra={"mode": recording.mode, "source": recording.source},
            )
            return None

        try:
            resolved_artifacts = resolver(resolution_context)
            if inspect.isawaitable(resolved_artifacts):
                resolved_artifacts = await resolved_artifacts
        except Exception:
            logger.exception(
                "failed to resolve recording artifacts for hamming payload",
                extra={"call_id": payload_record.get("call_id"), "mode": recording.mode},
            )
            return None

        if not isinstance(resolved_artifacts, dict):
            logger.warning(
                "recording resolver returned unsupported artifact payload",
                extra={"mode": recording.mode, "type": type(resolved_artifacts).__name__},
            )
            return None

        return resolved_artifacts

    def _apply_resolved_recording_artifacts(
        self,
        *,
        payload_record: dict[str, Any],
        resolved_artifacts: RecordingArtifacts,
    ) -> None:
        recording = self._runtime.config.recording
        if recording.mode == RECORDING_MODE_ROOM_COMPOSITE:
            self._apply_room_composite_recording_artifacts(
                payload_record=payload_record,
                resolved_artifacts=resolved_artifacts,
            )
            return

        self._apply_participant_egress_recording_artifacts(
            payload_record=payload_record,
            resolved_artifacts=resolved_artifacts,
        )

    def _apply_room_composite_recording_artifacts(
        self,
        *,
        payload_record: dict[str, Any],
        resolved_artifacts: RecordingArtifacts,
    ) -> None:
        recording_url = resolved_artifacts.get("recording_url")
        if isinstance(recording_url, str) and recording_url.strip():
            payload_record["recording_url"] = recording_url.strip()
            return

        logger.warning(
            "room_composite resolver did not return recording_url",
            extra={"call_id": payload_record.get("call_id")},
        )

    def _apply_participant_egress_recording_artifacts(
        self,
        *,
        payload_record: dict[str, Any],
        resolved_artifacts: RecordingArtifacts,
    ) -> None:
        recording_payload = payload_record.get("recording")
        if not isinstance(recording_payload, dict):
            recording_payload = {}
            payload_record["recording"] = recording_payload

        agent_recording_url = resolved_artifacts.get("agent_recording_url")
        user_recording_url = resolved_artifacts.get("user_recording_url")
        if isinstance(agent_recording_url, str) and agent_recording_url.strip():
            recording_payload["agent_recording_url"] = agent_recording_url.strip()
        if isinstance(user_recording_url, str) and user_recording_url.strip():
            recording_payload["user_recording_url"] = user_recording_url.strip()

        if (
            "agent_recording_url" not in recording_payload
            or "user_recording_url" not in recording_payload
        ):
            logger.warning(
                "participant_egress resolver did not return both recording URLs",
                extra={"call_id": payload_record.get("call_id")},
            )

    async def _attach_inline_recording_capture(
        self,
        *,
        report: SessionReportLike,
        payload_record: dict[str, Any],
    ) -> None:
        if not report.audio_recording_path:
            return

        try:
            recording_capture = await self._runtime.transport.build_inline_recording_capture(
                recording_path=report.audio_recording_path,
            )
        except Exception:
            logger.exception(
                "failed to capture inline recording for hamming payload",
                extra={"call_id": payload_record.get("call_id")},
            )
            return

        if recording_capture:
            payload_record["recording_capture"] = recording_capture

    async def _resolve_hamming_room_composite_recording_url(
        self,
        *,
        resolution_context: RecordingResolutionContext,
    ) -> str | None:
        test_case_run_id = _extract_test_case_run_id(
            participant_metadata_raw=resolution_context.participant_metadata_raw,
            recording_context=resolution_context.recording_context,
        )
        if not test_case_run_id:
            logger.info(
                "skipping hamming test-run room composite lookup because no test-case correlation id was found",
                extra={
                    "call_id": resolution_context.call_id,
                    "room_name": resolution_context.room_name,
                    **_recording_debug_context(
                        participant_metadata_raw=resolution_context.participant_metadata_raw,
                        recording_context=resolution_context.recording_context,
                    ),
                },
            )
            return None

        max_attempts = _env_int("HAMMING_ROOM_COMPOSITE_LOOKUP_ATTEMPTS", default=90)
        poll_interval_seconds = _env_float(
            "HAMMING_ROOM_COMPOSITE_LOOKUP_INTERVAL_SECONDS",
            default=1.0,
        )

        for attempt in range(1, max_attempts + 1):
            try:
                recording_url = await self._runtime.transport.fetch_test_case_run_recording_url(
                    test_case_run_id=test_case_run_id,
                )
            except Exception:
                logger.exception(
                    "failed to fetch room composite recording from hamming test run",
                    extra={
                        "test_case_run_id": test_case_run_id,
                        "call_id": resolution_context.call_id,
                        "attempt": attempt,
                    },
                )
                if attempt >= max_attempts:
                    return None
                await asyncio.sleep(poll_interval_seconds)
                continue

            if recording_url:
                logger.info(
                    "resolved room composite recording from hamming test run",
                    extra={
                        "test_case_run_id": test_case_run_id,
                        "call_id": resolution_context.call_id,
                        "attempt": attempt,
                    },
                )
                return recording_url

            if attempt < max_attempts:
                await asyncio.sleep(poll_interval_seconds)

        return None

    async def _start_plugin_managed_room_composite_egress(
        self,
        *,
        filepath: str,
    ) -> str:
        if self._job_ctx is None:
            raise RuntimeError("JobContext is required for plugin-managed room composite egress")

        egress_id = await self._runtime.transport.start_plugin_managed_room_composite_egress(
            room_name=self._job_ctx.room.name,
            filepath=filepath,
        )
        if self._recording_context is None:
            self._recording_context = cast(RecordingContext, {})
        self._recording_context["provider_recording_id"] = egress_id
        self._recording_context["provider_recording_filepath"] = filepath
        self._recording_context["room_name"] = self._job_ctx.room.name
        logger.info(
            "plugin-managed room composite egress started for session",
            extra={
                "room_name": self._job_ctx.room.name,
                "filepath": filepath,
                "egress_id": egress_id,
                **_recording_debug_context(
                    participant_metadata_raw=self._participant_metadata,
                    recording_context=self._recording_context,
                ),
            },
        )
        return egress_id

    async def _resolve_plugin_managed_room_composite_recording_url(
        self,
        *,
        resolution_context: RecordingResolutionContext,
    ) -> str | None:
        state = self._plugin_managed_room_composite
        if state is None:
            logger.info(
                "plugin-managed room composite state was not initialized for session",
                extra={
                    "call_id": resolution_context.call_id,
                    "room_name": resolution_context.room_name,
                    **_recording_debug_context(
                        participant_metadata_raw=resolution_context.participant_metadata_raw,
                        recording_context=resolution_context.recording_context,
                    ),
                },
            )
            return None

        try:
            egress_id = await state.start_task
        except Exception:
            logger.exception(
                "failed to start plugin-managed room composite egress",
                extra={"room_name": state.room_name, "filepath": state.filepath},
            )
            return None

        try:
            logger.info(
                "waiting for plugin-managed room composite egress recording URL",
                extra={
                    "egress_id": egress_id,
                    "room_name": state.room_name,
                    "filepath": state.filepath,
                    "call_id": resolution_context.call_id,
                    "max_attempts": _env_int(
                        "HAMMING_PLUGIN_MANAGED_ROOM_COMPOSITE_LOOKUP_ATTEMPTS",
                        default=90,
                    ),
                    "poll_interval_seconds": _env_float(
                        "HAMMING_PLUGIN_MANAGED_ROOM_COMPOSITE_LOOKUP_INTERVAL_SECONDS",
                        default=1.0,
                    ),
                },
            )
            recording_url = await self._runtime.transport.stop_plugin_managed_room_composite_egress_and_wait_for_url(
                egress_id=egress_id,
                filepath=state.filepath,
                max_attempts=_env_int(
                    "HAMMING_PLUGIN_MANAGED_ROOM_COMPOSITE_LOOKUP_ATTEMPTS",
                    default=90,
                ),
                poll_interval_seconds=_env_float(
                    "HAMMING_PLUGIN_MANAGED_ROOM_COMPOSITE_LOOKUP_INTERVAL_SECONDS",
                    default=1.0,
                ),
            )
        except Exception:
            logger.exception(
                "failed to finalize plugin-managed room composite egress",
                extra={
                    "egress_id": egress_id,
                    "room_name": state.room_name,
                    "call_id": resolution_context.call_id,
                },
            )
            return None

        if recording_url:
            logger.info(
                "resolved room composite recording from plugin-managed egress",
                extra={
                    "egress_id": egress_id,
                    "room_name": state.room_name,
                    "call_id": resolution_context.call_id,
                    "recording_url": recording_url,
                },
            )
        else:
            logger.warning(
                "plugin-managed room composite egress did not produce a recording URL",
                extra={
                    "egress_id": egress_id,
                    "room_name": state.room_name,
                    "call_id": resolution_context.call_id,
                    "filepath": state.filepath,
                },
            )
        return recording_url

    def _log_recording_resolution_outcome(
        self,
        *,
        report: SessionReportLike,
        payload_record: dict[str, Any],
    ) -> None:
        recording = self._runtime.config.recording
        recording_field_names = (
            "recording_url",
            "recording_capture",
        )
        nested_recording_field_names = (
            "agent_recording_url",
            "user_recording_url",
        )
        resolved_fields = [
            field_name for field_name in recording_field_names if payload_record.get(field_name)
        ]
        nested_recording = payload_record.get("recording")
        if isinstance(nested_recording, dict):
            resolved_fields.extend(
                field_name
                for field_name in nested_recording_field_names
                if nested_recording.get(field_name)
            )
        log_method = logger.info if resolved_fields else logger.warning
        log_method(
            "hamming recording payload resolution summary",
            extra={
                "call_id": payload_record.get("call_id"),
                "room_name": report.room,
                "recording_mode": recording.mode,
                "recording_source": recording.source,
                "resolved_fields": resolved_fields,
                **_recording_debug_context(
                    participant_metadata_raw=self._participant_metadata,
                    recording_context=self._recording_context,
                ),
            },
        )


class HammingRuntime:
    def __init__(self, config: HammingConfig) -> None:
        self.config = config
        self.transport = HammingTransport(
            base_url=config.base_url,
            api_key=config.api_key,
            policy=ConnectionPolicy(
                timeout_seconds=config.request_timeout_seconds,
                max_retries=config.max_retries,
                retry_backoff_seconds=config.retry_backoff_seconds,
            ),
        )
        self._monitors: dict[int, _SessionMonitor] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    def track_task(self, task: asyncio.Task[Any]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def remove_monitor(self, session_key: int) -> None:
        self._monitors.pop(session_key, None)
        if self._monitors:
            return

        close_task = asyncio.create_task(self.aclose())
        self.track_task(close_task)

    async def aclose(self) -> None:
        await self.transport.aclose()

    def attach_session(
        self,
        session: AgentSession[Any],
        *,
        job_ctx: JobContext | None = None,
        participant_identity: str | None = None,
        participant_metadata: str | None = None,
        external_agent_id: str | None = None,
        recording_context: RecordingContext | None = None,
    ) -> None:
        session_key = id(session)
        if session_key in self._monitors:
            return

        resolved_job_ctx = job_ctx or _safe_get_job_context()
        resolved_participant_identity = participant_identity or _participant_identity_from_job_ctx(
            resolved_job_ctx
        )
        resolved_participant_metadata = participant_metadata or _participant_metadata_from_job_ctx(
            resolved_job_ctx
        )
        resolved_external_agent_id = _resolve_external_agent_id(
            config=self.config,
            job_ctx=resolved_job_ctx,
            explicit_external_agent_id=external_agent_id,
        )

        if self.config.auto_record_audio:
            _wrap_session_start_for_recording(session)
        elif self.config.recording.mode == RECORDING_MODE_SESSION_AUDIO:
            _wrap_session_start_for_recording(session)

        monitor = _SessionMonitor(
            runtime=self,
            session=session,
            participant_identity=resolved_participant_identity,
            participant_metadata=resolved_participant_metadata,
            external_agent_id=resolved_external_agent_id,
            job_ctx=resolved_job_ctx,
            session_key=session_key,
            recording_context=recording_context,
        )

        session.on("close", monitor.on_close)
        if self.config.flush_on_shutdown and resolved_job_ctx is not None:
            resolved_job_ctx.add_shutdown_callback(monitor.on_shutdown)
        elif self.config.flush_on_shutdown:
            logger.warning(
                "flush_on_shutdown is enabled but JobContext is unavailable; final payload will "
                "flush only on session close"
            )

        monitor.start_plugin_managed_room_composite_if_needed()
        self._monitors[session_key] = monitor

    def build_report(
        self,
        session: AgentSession[Any],
        job_ctx: JobContext | None,
    ) -> SessionReportLike:
        active_job_ctx = job_ctx or _safe_get_job_context()
        if active_job_ctx is None:
            raise RuntimeError("JobContext not found; cannot build session report")

        make_session_report = getattr(active_job_ctx, "make_session_report", None)
        if not callable(make_session_report):
            raise RuntimeError(
                "JobContext.make_session_report() is unavailable; "
                "livekit-agents>=1.4.4 is required for Hamming monitoring exports"
            )

        return cast(SessionReportLike, make_session_report(session))


_RUNTIME: HammingRuntime | None = None


def configure_runtime(config: HammingConfig) -> HammingRuntime:
    global _RUNTIME

    if _RUNTIME is not None:
        logger.warning("hamming already configured, keeping existing runtime")
        return _RUNTIME

    _validate_config(config)
    _apply_runtime_environment_defaults(config)

    runtime = HammingRuntime(config)
    _RUNTIME = runtime
    return runtime


def attach_session(
    session: AgentSession[Any],
    *,
    job_ctx: JobContext | None = None,
    participant_identity: str | None = None,
    participant_metadata: str | None = None,
    external_agent_id: str | None = None,
    recording_context: RecordingContext | None = None,
) -> None:
    runtime = get_runtime()
    if runtime is None:
        raise RuntimeError("hamming is not configured. Call configure_hamming(...) first.")

    runtime.attach_session(
        session,
        job_ctx=job_ctx,
        participant_identity=participant_identity,
        participant_metadata=participant_metadata,
        external_agent_id=external_agent_id,
        recording_context=recording_context,
    )


def get_runtime() -> HammingRuntime | None:
    return _RUNTIME


def build_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    call_review_enabled: bool = True,
    flush_on_shutdown: bool = True,
    auto_record_audio: bool = False,
    external_agent_id: str | None = None,
    plugin_api_version: str = "1.0.0",
    payload_schema_version: str = "2026-03-02",
    call_id_strategy: str = CALL_ID_STRATEGY_ROOM_NAME,
    call_id_metadata_key: str = "call_id",
    resolve_call_id: CallIdResolver | None = None,
    recording_mode: str | None = None,
    sampling_rate: float = 1.0,
    sampling_key: str = "participant_identity",
    capture: CaptureConfigInput | None = None,
    redact: RedactConfigInput | None = None,
    livekit: LiveKitConfigInput | None = None,
    livekit_url: str | None = None,
    livekit_api_key: str | None = None,
    livekit_api_secret: str | None = None,
    include_interim_transcripts: bool | None = None,
    request_timeout_seconds: float = 10.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 0.5,
    recording: RecordingConfigInput | None = None,
    streaming: StreamingConfigInput | None = None,
) -> HammingConfig:
    resolved_api_key = api_key or os.getenv("HAMMING_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Hamming API key required. Pass api_key= or set HAMMING_API_KEY environment variable."
        )

    resolved_base_url = (
        base_url or os.getenv("HAMMING_BASE_URL") or "https://app.hamming.ai"
    ).rstrip("/")

    config = HammingConfig(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        call_review_enabled=call_review_enabled,
        flush_on_shutdown=flush_on_shutdown,
        auto_record_audio=auto_record_audio,
        external_agent_id=external_agent_id,
        plugin_api_version=plugin_api_version,
        payload_schema_version=payload_schema_version,
        call_id_strategy=_build_call_id_strategy(
            call_id_strategy=call_id_strategy,
            resolve_call_id=resolve_call_id,
        ),
        call_id_metadata_key=_normalize_optional_string(call_id_metadata_key) or "call_id",
        resolve_call_id=resolve_call_id,
        sampling=_build_sampling_config(rate=sampling_rate, key=sampling_key),
        capture=_build_capture_config(
            capture=capture,
            include_interim_transcripts=include_interim_transcripts,
        ),
        redact=_build_redact_config(redact=redact),
        livekit=_build_livekit_config(
            livekit=livekit,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
        ),
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        recording=_build_recording_config(
            recording=recording,
            auto_record_audio=auto_record_audio,
            recording_mode=recording_mode,
        ),
        streaming=_build_streaming_config(streaming=streaming),
    )
    _validate_config(config)
    return config


def _validate_config(config: HammingConfig) -> None:
    if not config.call_review_enabled:
        logger.warning("call_review_enabled is False; hamming payload export is disabled")

    _validate_payload_config(config)
    _validate_recording_config(config)
    _validate_streaming_config(config)


def _validate_payload_config(config: HammingConfig) -> None:
    if not config.plugin_api_version:
        raise ValueError("plugin_api_version is required")
    if not config.payload_schema_version:
        raise ValueError("payload_schema_version is required")
    if config.call_id_strategy not in SUPPORTED_CALL_ID_STRATEGIES:
        raise ValueError(
            "Unsupported call_id_strategy. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_CALL_ID_STRATEGIES))}."
        )
    if config.call_id_strategy == CALL_ID_STRATEGY_CUSTOM and config.resolve_call_id is None:
        raise ValueError("call_id_strategy='custom' requires resolve_call_id")


def _validate_recording_config(config: HammingConfig) -> None:
    recording = config.recording
    if recording.mode == RECORDING_MODE_PARTICIPANT_EGRESS:
        _validate_participant_egress_config(recording)
        return

    if recording.mode != RECORDING_MODE_ROOM_COMPOSITE:
        return

    if recording.delivery == RECORDING_DELIVERY_UPLOAD:
        raise ValueError(
            f"recording mode '{RECORDING_MODE_ROOM_COMPOSITE}' does not support delivery='upload' yet"
        )
    if recording.source == RECORDING_SOURCE_EXTERNAL_URLS and recording.resolver is None:
        raise ValueError(
            f"recording mode '{RECORDING_MODE_ROOM_COMPOSITE}' requires a callable "
            "recording resolver when source='external_urls'"
        )

    if recording.source == RECORDING_SOURCE_AUTO_EGRESS:
        missing_env_names = _missing_auto_egress_runtime_env_names(config)
        if missing_env_names:
            raise ValueError(
                "recording_mode='auto_egress' requires runtime configuration for: "
                f"{', '.join(missing_env_names)}"
            )


def _validate_participant_egress_config(recording: ResolvedRecordingConfig) -> None:
    if recording.source != RECORDING_SOURCE_EXTERNAL_URLS:
        raise ValueError(
            f"recording mode '{RECORDING_MODE_PARTICIPANT_EGRESS}' currently supports only "
            f"source='{RECORDING_SOURCE_EXTERNAL_URLS}'"
        )
    if recording.delivery == RECORDING_DELIVERY_UPLOAD:
        raise ValueError(
            f"recording mode '{RECORDING_MODE_PARTICIPANT_EGRESS}' does not support delivery='upload' yet"
        )
    if recording.resolver is None:
        raise ValueError(
            f"recording mode '{RECORDING_MODE_PARTICIPANT_EGRESS}' requires a callable recording resolver"
        )


def _validate_streaming_config(config: HammingConfig) -> None:
    if config.streaming.mode == STREAMING_MODE_TRACK_EGRESS:
        logger.info(
            "track_egress streaming is customer-managed; hamming plugin will not start streaming",
        )


def _safe_get_job_context() -> JobContext | None:
    try:
        return get_job_context()
    except RuntimeError:
        return None


def _participant_identity_from_job_ctx(job_ctx: JobContext | None) -> str | None:
    if job_ctx is None:
        return None

    for participant in job_ctx.room.remote_participants.values():
        identity = getattr(participant, "identity", None)
        if isinstance(identity, str) and identity:
            return identity

    identity = getattr(job_ctx, "local_participant_identity", None)
    if isinstance(identity, str) and identity:
        return identity

    agent = getattr(job_ctx, "agent", None)
    agent_identity = getattr(agent, "identity", None)
    if isinstance(agent_identity, str) and agent_identity:
        return agent_identity

    return None


def _participant_metadata_from_job_ctx(job_ctx: JobContext | None) -> str | None:
    if job_ctx is None:
        return None

    for participant in job_ctx.room.remote_participants.values():
        metadata = _serialize_participant_metadata(participant)
        if metadata:
            return metadata

    return None


def _serialize_participant_metadata(participant: object) -> str | None:
    metadata = getattr(participant, "metadata", None)
    attributes = getattr(participant, "attributes", None)
    parsed_metadata = _parse_participant_metadata_json(metadata)
    merged_metadata = _merge_participant_attributes(parsed_metadata, attributes)

    if merged_metadata:
        return json.dumps(merged_metadata)
    if parsed_metadata:
        return json.dumps(parsed_metadata)
    if isinstance(metadata, str) and metadata:
        return metadata
    return None


def _parse_participant_metadata_json(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, str) or not metadata:
        return {}

    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return parsed
    return {}


def _merge_participant_attributes(
    metadata: dict[str, Any],
    attributes: object,
) -> dict[str, Any]:
    if not isinstance(attributes, dict) or not attributes:
        return {}

    merged = {**metadata}
    for key, value in attributes.items():
        if isinstance(key, str):
            merged[key] = value
    return merged


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        return max(1, int(raw.strip()))
    except ValueError:
        return default


def _env_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        return max(0.1, float(raw.strip()))
    except ValueError:
        return default


def _env_optional_bool(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    return _env_flag(name, default=False)


def _default_auto_recording_options() -> dict[str, bool]:
    return {
        "audio": True,
        "traces": False,
        "logs": False,
        "transcript": False,
    }


def _build_recording_config(
    *,
    recording: RecordingConfigInput | None,
    auto_record_audio: bool,
    recording_mode: str | None,
) -> ResolvedRecordingConfig:
    if recording is None and recording_mode is not None:
        normalized_recording_mode = _normalize_recording_source(
            recording_mode,
            fallback=RECORDING_SOURCE_AUTO_EGRESS,
        )
        recording = cast(
            RecordingConfigInput,
            {
                "mode": RECORDING_MODE_ROOM_COMPOSITE,
                "source": normalized_recording_mode,
                "delivery": RECORDING_DELIVERY_URL,
            },
        )

    if recording is None:
        if auto_record_audio:
            return ResolvedRecordingConfig(mode=RECORDING_MODE_SESSION_AUDIO)
        return ResolvedRecordingConfig()

    raw_mode = _normalize_lower_string(recording.get("mode"), fallback=RECORDING_MODE_NONE)
    if raw_mode not in SUPPORTED_RECORDING_MODES:
        raise ValueError(
            "Unsupported recording mode. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_RECORDING_MODES))}."
        )

    default_source = (
        RECORDING_SOURCE_AUTO_EGRESS
        if raw_mode in {RECORDING_MODE_NONE, RECORDING_MODE_SESSION_AUDIO}
        else RECORDING_SOURCE_EXTERNAL_URLS
    )
    source = _normalize_recording_source(recording.get("source"), fallback=default_source)
    if source not in SUPPORTED_RECORDING_SOURCES:
        raise ValueError(
            "Unsupported recording source. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_RECORDING_SOURCES))}."
        )

    delivery = _normalize_lower_string(recording.get("delivery"), fallback=RECORDING_DELIVERY_AUTO)
    if delivery not in SUPPORTED_RECORDING_DELIVERIES:
        raise ValueError(
            "Unsupported recording delivery. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_RECORDING_DELIVERIES))}."
        )

    resolver = recording.get("resolver")
    if resolver is not None and not callable(resolver):
        raise ValueError("recording.resolver must be callable")

    return ResolvedRecordingConfig(
        mode=raw_mode,
        source=source,
        delivery=delivery,
        resolver=resolver,
    )


def _build_streaming_config(
    *,
    streaming: StreamingConfigInput | None,
) -> ResolvedStreamingConfig:
    if streaming is None:
        return ResolvedStreamingConfig(mode=STREAMING_MODE_NONE)

    mode = _normalize_lower_string(streaming.get("mode"), fallback=STREAMING_MODE_NONE)
    if mode not in SUPPORTED_STREAMING_MODES:
        raise ValueError(
            "Unsupported streaming mode. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_STREAMING_MODES))}."
        )

    source = _normalize_lower_string(
        streaming.get("source"),
        fallback=RECORDING_SOURCE_EXTERNAL_URLS,
    )
    if source == LEGACY_RECORDING_SOURCE_CUSTOMER_MANAGED:
        source = RECORDING_SOURCE_EXTERNAL_URLS
    elif source == LEGACY_RECORDING_SOURCE_PLUGIN_MANAGED:
        source = RECORDING_SOURCE_AUTO_EGRESS
    if source not in SUPPORTED_RECORDING_SOURCES:
        raise ValueError(
            "Unsupported streaming source. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_RECORDING_SOURCES))}."
        )

    return ResolvedStreamingConfig(mode=mode, source=source)


def _build_call_id_strategy(
    *,
    call_id_strategy: str,
    resolve_call_id: CallIdResolver | None,
) -> str:
    normalized_strategy = _normalize_lower_string(
        call_id_strategy,
        fallback=CALL_ID_STRATEGY_ROOM_NAME,
    )
    if normalized_strategy not in SUPPORTED_CALL_ID_STRATEGIES:
        raise ValueError(
            "Unsupported call_id_strategy. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_CALL_ID_STRATEGIES))}."
        )
    if normalized_strategy == CALL_ID_STRATEGY_CUSTOM and resolve_call_id is None:
        raise ValueError("call_id_strategy='custom' requires resolve_call_id")
    return normalized_strategy


def _build_capture_config(
    *,
    capture: CaptureConfigInput | None,
    include_interim_transcripts: bool | None,
) -> ResolvedCaptureConfig:
    capture = capture or {}
    prompts = _normalize_lower_string(capture.get("prompts"), fallback="full")
    tools = _normalize_lower_string(capture.get("tools"), fallback="full")
    if prompts not in SUPPORTED_CAPTURE_MODES:
        raise ValueError("capture.prompts must be one of: full, none")
    if tools not in SUPPORTED_CAPTURE_MODES:
        raise ValueError("capture.tools must be one of: full, none")

    interim_transcripts = capture.get("interim_transcripts")
    if include_interim_transcripts is not None:
        interim_transcripts = include_interim_transcripts

    return ResolvedCaptureConfig(
        agent_context=bool(capture.get("agent_context", True)),
        prompts=prompts,
        tools=tools,
        interim_transcripts=bool(interim_transcripts),
    )


def _build_redact_config(*, redact: RedactConfigInput | None) -> ResolvedRedactConfig:
    redact = redact or {}
    return ResolvedRedactConfig(
        pii=bool(redact.get("pii", True)),
        tool_args_paths=_normalize_string_list(redact.get("tool_args_paths")),
        tool_output_paths=_normalize_string_list(redact.get("tool_output_paths")),
    )


def _build_livekit_config(
    *,
    livekit: LiveKitConfigInput | None,
    livekit_url: str | None,
    livekit_api_key: str | None,
    livekit_api_secret: str | None,
) -> ResolvedLiveKitConfig:
    livekit = livekit or {}
    return ResolvedLiveKitConfig(
        url=_normalize_optional_string(livekit.get("url") or livekit_url),
        api_key=_normalize_optional_string(livekit.get("api_key") or livekit_api_key),
        api_secret=_normalize_optional_string(livekit.get("api_secret") or livekit_api_secret),
    )


def _build_sampling_config(*, rate: float, key: str) -> ResolvedSamplingConfig:
    if rate < 0 or rate > 1:
        raise ValueError("sampling_rate must be between 0.0 and 1.0")
    normalized_key = _normalize_lower_string(key, fallback="participant_identity")
    return ResolvedSamplingConfig(rate=rate, key=normalized_key)


def _wrap_session_start_for_recording(session: AgentSession[Any]) -> None:
    if getattr(session, "_hamming_auto_record_wrapped", False):
        return

    original_start = session.start

    async def _wrapped_start(self: AgentSession[Any], *args: Any, **kwargs: Any) -> Any:
        record = kwargs.get("record", NOT_GIVEN)
        if record is NOT_GIVEN or record is None:
            kwargs["record"] = _default_auto_recording_options()
        if inspect.ismethod(original_start):
            return await cast(Callable[..., Awaitable[Any]], original_start)(*args, **kwargs)
        return await cast(Callable[..., Awaitable[Any]], original_start)(self, *args, **kwargs)

    session_runtime = cast(Any, session)
    session_runtime.start = MethodType(_wrapped_start, session)
    session_runtime._hamming_auto_record_wrapped = True


def _resolve_external_agent_id(
    *,
    config: HammingConfig,
    job_ctx: JobContext | None,
    explicit_external_agent_id: str | None,
) -> str:
    if explicit_external_agent_id:
        return explicit_external_agent_id

    if config.external_agent_id:
        return config.external_agent_id

    env_value = os.getenv("HAMMING_EXTERNAL_AGENT_ID")
    if env_value:
        return env_value

    if job_ctx is not None:
        agent_name = job_ctx.job.agent_name
        if agent_name:
            return agent_name

    raise ValueError(
        "external_agent_id is required. Pass external_agent_id= to configure_hamming(...) or "
        "attach_session(...), set HAMMING_EXTERNAL_AGENT_ID, or ensure job.agent_name is set."
    )


def _normalize_lower_string(value: object, *, fallback: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            return normalized
    return fallback


def _normalize_optional_string(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _normalize_string_list(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    normalized_values: list[str] = []
    for value in values:
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                normalized_values.append(candidate)
    return tuple(normalized_values)


def _normalize_recording_source(value: object, *, fallback: str) -> str:
    normalized = _normalize_lower_string(value, fallback=fallback)
    if normalized == LEGACY_RECORDING_SOURCE_CUSTOMER_MANAGED:
        return RECORDING_SOURCE_EXTERNAL_URLS
    if normalized == LEGACY_RECORDING_SOURCE_PLUGIN_MANAGED:
        return RECORDING_SOURCE_AUTO_EGRESS
    return normalized


def _build_capture_manifest(config: HammingConfig) -> dict[str, Any]:
    return {
        "agent_context": config.capture.agent_context,
        "prompts": config.capture.prompts,
        "tools": config.capture.tools,
        "interim_transcripts": config.capture.interim_transcripts,
        "recording_mode": config.recording.source,
        "recording_shape": config.recording.mode,
        "fallback_behavior": (
            "resolver_required"
            if config.recording.source == RECORDING_SOURCE_EXTERNAL_URLS
            else "plugin_managed_egress"
        ),
    }


def _build_sampling_metadata(
    *,
    config: HammingConfig,
    call_id: str,
    room_name: str,
    participant_identity: str | None,
    participant_metadata_raw: str | None,
    external_agent_id: str,
) -> dict[str, Any]:
    sample_key_value = _resolve_sampling_key_value(
        key=config.sampling.key,
        call_id=call_id,
        room_name=room_name,
        participant_identity=participant_identity,
        participant_metadata_raw=participant_metadata_raw,
        external_agent_id=external_agent_id,
    )
    hash_input = f"{SAMPLING_VERSION}:{config.sampling.key}:{sample_key_value}"
    hash_hex = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    hash_value = int(hash_hex, 16)
    threshold = int(config.sampling.rate * ((1 << 256) - 1))
    decision = "included" if hash_value <= threshold else "excluded"
    return {
        "rate": config.sampling.rate,
        "key": config.sampling.key,
        "decision": decision,
        "hash": hash_hex[:8],
        "version": SAMPLING_VERSION,
    }


def _resolve_sampling_key_value(
    *,
    key: str,
    call_id: str,
    room_name: str,
    participant_identity: str | None,
    participant_metadata_raw: str | None,
    external_agent_id: str,
) -> str:
    participant_metadata = _parse_json_object(participant_metadata_raw)
    if key in {"participant_identity", "identity"} and participant_identity:
        return participant_identity
    if key in {"call_id", "call"}:
        return call_id
    if key in {"room_name", "livekit_room_name"}:
        return room_name
    if key == "external_agent_id":
        return external_agent_id
    if key.startswith("participant_metadata."):
        metadata_key = key.split(".", 1)[1]
        value = participant_metadata.get(metadata_key)
        if value is not None:
            return str(value)
    return call_id


def _apply_runtime_environment_defaults(config: HammingConfig) -> None:
    env_defaults = {
        "LIVEKIT_URL": config.livekit.url,
        "LIVEKIT_API_KEY": config.livekit.api_key,
        "LIVEKIT_API_SECRET": config.livekit.api_secret,
    }
    for env_name, env_value in env_defaults.items():
        if env_value and not os.getenv(env_name):
            os.environ[env_name] = env_value


def _missing_auto_egress_runtime_env_names(config: HammingConfig) -> list[str]:
    available_values = {
        "LIVEKIT_URL": config.livekit.url or os.getenv("LIVEKIT_URL", "").strip() or None,
        "LIVEKIT_API_KEY": config.livekit.api_key
        or os.getenv("LIVEKIT_API_KEY", "").strip()
        or None,
        "LIVEKIT_API_SECRET": config.livekit.api_secret
        or os.getenv("LIVEKIT_API_SECRET", "").strip()
        or None,
        "AWS_RECORDINGS_BUCKET": os.getenv("AWS_RECORDINGS_BUCKET", "").strip() or None,
        "AWS_REGION": os.getenv("AWS_REGION", "").strip() or None,
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "").strip() or None,
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "").strip() or None,
    }
    return [name for name, value in available_values.items() if not value]


def _build_plugin_managed_room_composite_filepath(
    *,
    room_name: str,
    participant_metadata_raw: str | None,
    recording_context: RecordingContext | None,
    session_key: int,
) -> str:
    participant_metadata = _parse_json_object(participant_metadata_raw)
    call_id = participant_metadata.get("call_id")
    candidate = _extract_test_case_run_id(
        participant_metadata_raw=participant_metadata_raw,
        recording_context=recording_context,
    )
    if candidate is None and isinstance(call_id, str) and call_id.strip():
        candidate = call_id.strip()
    if candidate is None:
        candidate = room_name

    safe_candidate = _sanitize_recording_path_token(candidate)
    return (
        f"{PLUGIN_MANAGED_RECORDINGS_DIR}/"
        f"{PLUGIN_MANAGED_ROOM_COMPOSITE_FILENAME_PREFIX}_{safe_candidate}_{session_key}."
        f"{PLUGIN_MANAGED_ROOM_COMPOSITE_EXTENSION}"
    )


def _sanitize_recording_path_token(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    sanitized = sanitized.strip("-")
    return sanitized or "session"


def _extract_test_case_run_id(
    *,
    participant_metadata_raw: str | None,
    recording_context: RecordingContext | None,
) -> str | None:
    participant_metadata = _parse_json_object(participant_metadata_raw)
    for key in TEST_CASE_RUN_ID_METADATA_KEYS:
        value = participant_metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    if recording_context:
        customer_conversation_id = recording_context.get("customer_conversation_id")
        if isinstance(customer_conversation_id, str) and customer_conversation_id.strip():
            return customer_conversation_id.strip()

    return None


def _recording_debug_context(
    *,
    participant_metadata_raw: str | None,
    recording_context: RecordingContext | None,
) -> dict[str, Any]:
    participant_metadata = _parse_json_object(participant_metadata_raw)
    metadata_keys = sorted(key for key in participant_metadata.keys() if isinstance(key, str))
    return {
        "test_case_run_id": _extract_test_case_run_id(
            participant_metadata_raw=participant_metadata_raw,
            recording_context=recording_context,
        ),
        "participant_metadata_keys": metadata_keys,
        "customer_conversation_id": (
            recording_context.get("customer_conversation_id") if recording_context else None
        ),
        "provider_recording_id": (
            recording_context.get("provider_recording_id") if recording_context else None
        ),
        "provider_recording_filepath": (
            recording_context.get("provider_recording_filepath") if recording_context else None
        ),
        "recording_context_room_name": (
            recording_context.get("room_name") if recording_context else None
        ),
    }


def _parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return parsed

    return {}


def _string_or_fallback(value: object, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _reset_runtime_for_tests() -> None:
    global _RUNTIME
    runtime = _RUNTIME
    _RUNTIME = None

    if runtime is None:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(runtime.aclose())
        return

    loop.create_task(runtime.aclose())
