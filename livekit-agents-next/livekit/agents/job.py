# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import contextvars
import datetime
import functools
import inspect
import logging
import multiprocessing as mp
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field, replace
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import aiohttp

import livekit.rtc as rtc
from livekit import api
from livekit.api.access_token import Claims
from livekit.protocol import agent
from livekit.protocol.sip import TransferSIPParticipantResponse

from . import _core
from ._generated.telemetry import RecordingOptions as RecordingOptionsRecord, SessionReportSnapshot
from .log import logger
from .telemetry import _upload_session_report, trace_types, utils as telemetry_utils
from .telemetry._destination import _BufferingHandler, _JobTelemetry
from .telemetry._export import attribute_bytes
from .telemetry.metrics import flush_turn_metrics
from .telemetry.traces import (
    _cloud_log_handler,
    _discard_cloud_tracer,
    _prepare_cloud_tracer,
    _serialize_session_options,
    _setup_cloud_tracer,
    _shutdown_telemetry,
    capture_content_enabled,
    install_rpc_tracing,
    tracer,
)
from .types import (
    ATTRIBUTE_REDACTION_ENABLED,
    ATTRIBUTE_SIMULATION_ENABLED,
    ATTRIBUTE_SIMULATION_JOB_ID,
    ATTRIBUTE_SIMULATION_RUN_ID,
    ATTRIBUTE_SIMULATOR_DISPATCH,
    NotGivenOr,
)
from .utils import http_context, is_given, wait_for_participant
from .version import __version__

_JobContextVar = contextvars.ContextVar["JobContext"]("agents_job_context")


if TYPE_CHECKING:
    from ._generated.telemetry import ProjectedReportEvent
    from .llm import ChatContext
    from .metrics import ModelUsage
    from .simulation import SimulationContext
    from .storage.resolve import Thread
    from .voice.agent_session import RecordingOptions, VoiceSession
    from .voice.events import AgentEvent


@overload
def get_job_context(*, required: Literal[True] = True) -> JobContext: ...


@overload
def get_job_context(*, required: Literal[False]) -> JobContext | None: ...


def get_job_context(*, required: bool = True) -> JobContext | None:
    ctx = _JobContextVar.get(None)
    if ctx is None and required:
        raise RuntimeError(
            "no job context found, are you running this code inside a job entrypoint?"
        )

    return ctx


get_current_job_context = get_job_context


def _server_setting(name: str) -> str | None:
    """The server setting ``name`` (``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, ...): in a job, the
    value its ``AgentServer`` runs with (an option given in code, else the worker's environment);
    elsewhere, the environment variable."""
    ctx = get_job_context(required=False)
    if ctx is not None:
        return ctx._server_env.get(name)
    return os.environ.get(name)


def _server_credentials(
    url: NotGivenOr[str | None] = None,
    api_key: NotGivenOr[str | None] = None,
    api_secret: NotGivenOr[str | None] = None,
) -> tuple[str, str, str]:
    """Each given value, else the server's ``LIVEKIT_URL``/``API_KEY``/``API_SECRET``; ``""``
    when neither has one."""
    return (
        url or _server_setting("LIVEKIT_URL") or "",
        api_key or _server_setting("LIVEKIT_API_KEY") or "",
        api_secret or _server_setting("LIVEKIT_API_SECRET") or "",
    )


def current_simulation() -> SimulationContext | None:
    """The :class:`SimulationContext` of the job running on this task, or ``None``.

    ``None`` covers everything that is not a simulation: a production job, and code
    running outside a job context at all (console mode, tests). Unlike
    :attr:`JobContext.simulation_context` this does not need the job context in hand,
    so it can be called from deep inside the stack.
    """
    ctx = get_job_context(required=False)
    if ctx is None:
        return None

    return ctx.simulation_context


@unique
class JobExecutorType(Enum):
    PROCESS = "process"
    THREAD = "thread"


class AutoSubscribe(str, Enum):
    SUBSCRIBE_ALL = "subscribe_all"
    SUBSCRIBE_NONE = "subscribe_none"
    AUDIO_ONLY = "audio_only"
    VIDEO_ONLY = "video_only"


@dataclass
class JobAcceptArguments:
    name: str
    identity: str
    metadata: str
    attributes: dict[str, str] | None = None


@dataclass
class RunningJobInfo:
    accept_arguments: JobAcceptArguments
    job: agent.Job
    url: str
    token: str
    worker_id: str
    fake_job: bool
    # dispatch timeline, unix seconds; 0.0 when unknown (simulation, console, resumed jobs)
    received_at: float = 0.0
    """The worker received the availability request."""
    accepted_at: float = 0.0
    """The request handler accepted the job."""
    assigned_at: float = 0.0
    """The server's assignment (room token) arrived."""
    launched_at: float = 0.0
    """A process was acquired from the pool and handed the job."""


DEFAULT_PARTICIPANT_KINDS: list[rtc.ParticipantKind.ValueType] = [
    rtc.ParticipantKind.PARTICIPANT_KIND_CONNECTOR,
    rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
    rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
]


class _ContextLogFieldsFilter(logging.Filter):
    """Filter that adds job context fields to log records without overwriting."""

    def __init__(self, job_ctx: JobContext) -> None:
        super().__init__()
        self.job_ctx = job_ctx

    def filter(self, record: logging.LogRecord) -> bool:
        # only add fields for the current job context
        if self.job_ctx.proc.executor_type != JobExecutorType.PROCESS:
            try:
                ctx = get_job_context()
            except RuntimeError:
                return True
            else:
                if ctx != self.job_ctx:
                    return True

        # add context fields only if they don't already exist in the record
        for key, value in self.job_ctx._log_fields.items():
            if not hasattr(record, key):
                setattr(record, key, value)

        return True


def callable_name(callable: Any) -> str:
    """A callable's name, as the shutdown spans and logs label it."""
    for attr in ("__qualname__", "__name__"):
        if isinstance(name := getattr(callable, attr, None), str):
            return name
    return repr(callable)


class JobContext:
    _PARTICIPANT_ENTRYPOINT_CALLBACK = Callable[
        ["JobContext", rtc.RemoteParticipant], Coroutine[None, None, None]
    ]

    # private ctor
    def __init__(
        self,
        *,
        proc: JobProcess,
        info: RunningJobInfo,
        room: rtc.Room,
        run: _core.JobRun,
        server_env: dict[str, str],
    ) -> None:
        self._proc = proc
        # the server's settings by variable name (``AgentServer(ws_url=, api_key=, ...)`` or the
        # worker's environment): a job thread reads options given in code only from here
        self._server_env = server_env
        self._info = info
        self._room = room
        # the participant entrypoints, the connection, the API calls and the recording's setup
        self._run = run
        self._participant_entrypoints: list[Callable[..., Any]] = []
        # the job's primary session: the first started for it, which the job closes and reports
        self._primary_session: VoiceSession | None = None
        self._room.on("participant_connected", self._participant_available)

        self._log_fields: dict[str, Any] = {}
        self._log_filter = _ContextLogFieldsFilter(self)
        self._handlers_with_filter: list[logging.Handler] = []

        self._thread: Thread | None = None

        # on_simulation_end callback, injected by the job runner from AgentServer.
        self._simulation_end_fnc: Callable[[SimulationContext], Any] | None = None

        self._tempdir = tempfile.TemporaryDirectory()

        self._session_directory = Path(self._tempdir.name)

        self._tagger = _core.Tagger()
        self._early_log_handler: _BufferingHandler | None = None
        # this job's cloud-telemetry registration, set by init_recording():
        # span/log/metric attribution and upload gating resolve from it (the
        # OTel providers are shared across possibly-concurrent jobs). None while
        # the job has no registration to release.
        self._telemetry_state: _JobTelemetry | None = None

    def _on_setup(self) -> None:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(self._log_filter)
            self._handlers_with_filter.append(handler)

    def _start_log_buffering(self) -> None:
        """Start buffering logs early so crash logs can be uploaded."""
        if not self._run.buffer_logs:
            return

        self._early_log_handler = _BufferingHandler()
        logging.getLogger().addHandler(self._early_log_handler)

    def _prepare_telemetry(self) -> None:
        """Have the cloud trace pipeline up before the job's first span; whether anything is
        uploaded is decided in ``init_recording``, and the job's spans are held until then."""
        if self._info.fake_job:
            return
        if not (obs_url := self._run.observability_url):
            return
        try:
            _prepare_cloud_tracer(
                room_id=self.job.room.sid,
                job_id=self.job.id,
                agent_name=self.job.agent_name,
                observability_url=obs_url,
            )
        except Exception:
            logger.exception("failed to prepare the cloud trace pipeline")

    def _stop_log_buffering(self) -> None:
        """Remove the buffering handler without replaying."""
        handler = self._early_log_handler
        if handler is None:
            return
        logging.getLogger().removeHandler(handler)
        self._early_log_handler = None

    def _flush_early_log_buffer(self, *, replay: bool) -> None:
        """Remove buffering handler and optionally replay records through OTLP."""
        handler = self._early_log_handler
        if handler is None:
            return

        logging.getLogger().removeHandler(handler)
        self._early_log_handler = None

        if not replay:
            return

        # replay through the framework's own OTLP handler that _setup_cloud_tracer
        # just attached — the integrator may have their own OTel LoggingHandler on
        # the root logger, and the buffered records must not be routed into it
        otlp_handler = _cloud_log_handler()
        if otlp_handler is not None and otlp_handler in logging.getLogger().handlers:
            for record in handler.buffer:
                otlp_handler.emit(record)

    async def _on_session_end(self) -> None:
        if not (session := self._primary_agent_session):
            return

        if (finished := session._recording_finished) is not None:
            try:
                await finished
            except Exception:
                logger.exception("failed to finish the session recording")

        session = self._report_session(session)
        loop = asyncio.get_running_loop()
        preparation_deadline = loop.time() + _core.UPLOAD_TIMEOUT_SECONDS

        def prepare_report() -> SessionReport:
            from .llm import ChatContext

            return self._build_session_report(
                session,
                ChatContext(items=list(session.history.items)),
                session._report_event_snapshots(),
            )

        try:
            async with asyncio.timeout_at(preparation_deadline):
                report = await asyncio.to_thread(prepare_report)
        except Exception:
            logger.exception("failed to prepare the session report")
            return

        from .llm import ChatContext
        from .voice.remote_session import ConsoleCall

        try:
            async with asyncio.timeout_at(preparation_deadline):
                for message in report.chat_history.messages():
                    await asyncio.to_thread(flush_turn_metrics, ChatContext(items=[message]))
        except Exception:
            logger.exception("failed to record session metrics")

        if ConsoleCall.record and loop.time() < preparation_deadline:
            try:
                async with asyncio.timeout_at(preparation_deadline):
                    projection = _core._core_SessionReportProjection(
                        report._capture_content, report._recording_options, self._redaction_enabled
                    )
                    builder = _core._core_PreparedTranscript()
                    if projection.transcript:
                        from .telemetry.traces import _append_transcript_item

                        items = report.chat_history.items
                        builder.check_input(len(items), 0)
                        for item in items:
                            await asyncio.to_thread(
                                _append_transcript_item, builder, item, len(items)
                            )
                    snapshot, _ = await report._snapshot(
                        projection,
                        deadline=preparation_deadline,
                        event_sink=builder.append_event,
                    )
                    path = Path(
                        "console-recordings",
                        f"session-{datetime.datetime.now().strftime('%m-%d-%H%M%S')}",
                        "session_report.json",
                    )
                    await builder.finish_report(snapshot, str(path))
            except Exception:
                logger.exception("failed to save session report")

        has_evals = self._tagger.has_evals()
        obs_url = self._run.observability_url
        recording = _core.recording_enabled(RecordingOptionsRecord(**session._recording_options))
        if (recording or has_evals) and obs_url:
            try:
                await _upload_session_report(
                    agent_name=self._info.job.agent_name,
                    observability_url=obs_url,
                    report=report,
                    tagger=self._tagger,
                    http_session=http_context.http_session(),
                    metadata=self._otel_metadata(
                        session._recording_options,
                        redaction_enabled=self._redaction_enabled,
                    ),
                )
            except Exception:
                logger.exception("failed to upload the session report to LiveKit Cloud")

    async def _on_cleanup(self) -> None:
        # if session.start() was never reached and server wanted recording,
        # set up OTLP now and flush buffered crash logs
        if self._early_log_handler is not None and not self._run.recording_initialized:
            try:
                from .voice.agent_session import RecordingOptions

                self.init_recording(
                    RecordingOptions(audio=False, traces=False, logs=True, transcript=False)
                )
            except Exception:
                logger.exception("failed to initialize crash log upload")
                self._stop_log_buffering()

        def _cleanup_blocking() -> None:
            self._tempdir.cleanup()
            # per job: flushes this job's telemetry, leaves a concurrent job's untouched
            if self._telemetry_state is not None:
                _shutdown_telemetry(self.job.id)
            # a job that never registered still had spans held for it by the gate
            _discard_cloud_tracer(self.job.id)

        # exporter flushes and the tempdir cleanup block; keep them off the event loop
        await asyncio.to_thread(_cleanup_blocking)

        for handler in self._handlers_with_filter:
            handler.removeFilter(self._log_filter)
        self._handlers_with_filter.clear()

    def is_fake_job(self) -> bool:
        return self._info.fake_job

    @property
    def _primary_agent_session(self) -> VoiceSession | None:
        return self._primary_session

    def _claim_session(self, session: VoiceSession) -> bool:
        """A session started for the job: whether it is the job's primary one."""
        if self._primary_session is None:
            self._primary_session = session
        return self._primary_session is session

    @property
    def _redaction_enabled(self) -> bool:
        return self._run.redaction

    @property
    def session_directory(self) -> Path:
        return Path(self._session_directory)

    @property
    def tagger(self) -> _core.Tagger:
        """Returns the Tagger for adding tags and outcomes to the session.

        Tags are uploaded to LiveKit Cloud at session end.

        Example:
            ```python
            ctx.tagger.success(reason="Task completed successfully")
            ctx.tagger.fail(reason="User hung up before completing")
            ctx.tagger.add("voicemail:true")
            ```
        """
        return self._tagger

    def _report_session(self, session: VoiceSession | None) -> VoiceSession:
        session = session or self._primary_agent_session
        if session is None:
            raise RuntimeError("Cannot prepare report, no VoiceSession was found")
        if not session._event_retention.allows_read(self.job.id):
            raise PermissionError("A session report is available only to its owning job")
        return session

    def make_session_report(self, session: VoiceSession | None = None) -> SessionReport:
        session = self._report_session(session)
        return self._build_session_report(session, session.history.copy(), session._recorded_events)

    def _build_session_report(
        self, session: VoiceSession, history: ChatContext, events: list[AgentEvent]
    ) -> SessionReport:
        recording_started_at = session._recorder.started_at if session._recorder else None
        sr = SessionReport(
            job_id=self.job.id,
            room_id=self.job.room.sid,
            room=self.job.room.name,
            options={**session.options, "recording_options": session._recording_options},
            started_at=session._started_at,
            events=events,
            chat_history=history,
            model_usage=session.usage.model_usage,
            audio_recording_path=session._recording_path,
            audio_recording_started_at=recording_started_at,
            _capture_content=session._event_retention.capture_content,
        )
        if recording_started_at is not None:
            sr.duration = sr.timestamp - recording_started_at

        return sr

    @functools.cached_property
    def api(self) -> api.LiveKitAPI:
        """Returns an LiveKitAPI for making API calls to LiveKit, with the server's URL and key
        pair: the values passed to `AgentServer`, else LIVEKIT_URL, LIVEKIT_API_KEY and
        LIVEKIT_API_SECRET.
        """
        env = self._server_env
        return api.LiveKitAPI(
            env.get("LIVEKIT_URL"),
            env.get("LIVEKIT_API_KEY"),
            env.get("LIVEKIT_API_SECRET"),
            session=http_context.http_session(),
        )

    @property
    def proc(self) -> JobProcess:
        """Returns the process running the job. Useful for storing process-specific state."""
        return self._proc

    @property
    def job(self) -> agent.Job:
        """Returns the current job that the worker is executing."""
        return self._info.job

    @property
    def worker_id(self) -> str:
        """Returns the id of the worker."""
        return self._info.worker_id

    @property
    def thread(self) -> Thread | None:
        """The call's durable thread, resolved before the entrypoint runs: the ``thread_id`` of
        the dispatch metadata, else a new thread. ``None`` on a server without storage. Pass it
        to ``VoiceSession(thread=)``."""
        return self._thread

    @property
    def room(self) -> rtc.Room:
        """The Room object is the main interface that the worker should interact with.

        When the entrypoint is called, the worker has not connected to the Room yet.
        Certain properties of Room would not be available before calling JobContext.connect()
        """
        return self._room

    @property
    def agent(self) -> rtc.LocalParticipant:
        return self._room.local_participant

    @property
    def primary_session(self) -> VoiceSession:
        """Returns the primary VoiceSession for this job."""
        if not self._primary_agent_session:
            raise RuntimeError("No VoiceSession was started for this job")
        return self._primary_agent_session

    @functools.cached_property
    def simulation_context(self) -> SimulationContext | None:
        """The :class:`SimulationContext` when this job runs under a simulation, ``None`` for
        an ordinary job.

        Read from the job's ``lk.simulator.dispatch`` attribute (a protojson
        ``SimulationDispatch``), so it is final as soon as the entrypoint runs.
        """
        # The simulation attributes ride the agent dispatch and land on the job
        # itself, so this is final before the room even connects.
        metadata = self._info.job.attributes.get(ATTRIBUTE_SIMULATOR_DISPATCH, "")
        if not metadata:
            return None

        from google.protobuf import json_format

        from livekit.protocol import agent_simulation as sim_pb

        from .simulation import SimulationContext

        try:
            # ignore unknown fields so dispatches from newer servers still parse
            dispatch = json_format.Parse(
                metadata, sim_pb.SimulationDispatch(), ignore_unknown_fields=True
            )
        except json_format.ParseError:
            return None

        if not dispatch.simulation_run_id:
            return None

        return SimulationContext(dispatch, self)

    @property
    def inference_headers(self) -> dict[str, str]:
        """Extra headers this job puts on every LiveKit Inference request it makes.

        Merged last by ``inference.get_inference_headers``, so what the job asserts
        about itself outranks what an individual model was configured with. Empty for
        an ordinary job.
        """
        from .inference._utils import HEADER_INFERENCE_PRIORITY

        headers: dict[str, str] = {}

        # A text simulation is batch load: a run fans out many jobs at once and nobody
        # is waiting on the answers, so it must not compete with live traffic for
        # gateway capacity, and it must not be able to ask for priority either. Audio
        # simulations are excluded: they run in real time against the audio pipeline,
        # so their latency has to stay representative of production.
        sim = self.simulation_context
        if sim is not None and sim.simulation_mode == "text":
            headers[HEADER_INFERENCE_PRIORITY] = "low"

        return headers

    @property
    def local_participant_identity(self) -> str:
        if identity := self.token_claims().identity:
            return identity

        return self._room.local_participant.identity

    @property
    def log_context_fields(self) -> dict[str, Any]:
        """
        Returns the current dictionary of log fields that will be injected into log records.

        These fields enable enriched structured logging and can include job metadata,
        worker ID, trace IDs, or other diagnostic context.

        The returned dictionary can be directly edited, or entirely replaced via assignment
        (e.g., `job_context.log_context_fields = {...}`)
        """
        return self._log_fields

    @log_context_fields.setter
    def log_context_fields(self, fields: dict[str, Any]) -> None:
        """
        Sets the log fields to be injected into future log records.

        Args:
            fields (dict[str, Any]): A dictionary of key-value pairs representing
                structured data to attach to each log entry. Typically includes contextual
                information like job ID, trace information, or worker metadata.
        """
        self._log_fields = fields

    def add_shutdown_callback(
        self,
        callback: Callable[[], Coroutine[None, None, None]]
        | Callable[[str], Coroutine[None, None, None]],
    ) -> None:
        """
        Add a callback to be called when the job is shutting down.
        Optionally the callback can take a single argument, the shutdown reason.
        """
        bound = 1 if inspect.ismethod(callback) else 0
        module = getattr(callback, "__module__", "") or ""
        self._run.add_shutdown_callback(
            callback,
            takes_reason=cast(Any, callback).__code__.co_argcount > bound,
            framework=module == "livekit.agents" or module.startswith("livekit.agents."),
            name=callable_name(callback),
        )

    async def wait_for_participant(
        self,
        *,
        identity: str | None = None,
        kind: list[rtc.ParticipantKind.ValueType]
        | rtc.ParticipantKind.ValueType = DEFAULT_PARTICIPANT_KINDS,
    ) -> rtc.RemoteParticipant:
        """
        Returns a participant that matches the given identity. If identity is None, the first
        participant that joins the room will be returned.
        If the participant has already joined, the function will return immediately.
        """

        # handle connection automatically, otherwise wait_for_participant will raise an error
        if not self._room.isconnected():
            await self.connect()

        with tracer.detached_span(
            "wait_for_participant",
            attributes={trace_types.ATTR_ROOM_IO_PARTICIPANT_FILTER: identity is not None},
        ) as span:
            participant = await wait_for_participant(self._room, identity=identity, kind=kind)
            span.set_attributes(telemetry_utils.participant_attributes(participant))
            return participant

    async def connect(
        self,
        *,
        encryption: rtc.E2EEOptions | None = None,
        auto_subscribe: AutoSubscribe = AutoSubscribe.SUBSCRIBE_ALL,
        rtc_config: rtc.RtcConfiguration | None = None,
        single_peer_connection: bool | None = None,
    ) -> None:
        """Connect to the room. This method should be called only once.

        Args:
            encryption: End-to-end encryption options. If provided, the Agent will utilize end-to-end encryption. Note: clients will also need to handle E2EE.
            auto_subscribe: Whether to automatically subscribe to tracks. Default is AutoSubscribe.SUBSCRIBE_ALL.
            rtc_config: Custom RTC configuration to use when connecting to the room.
            single_peer_connection: Use a single peer connection for both publish and subscribe. When None, uses the default (False).
        """  # noqa: E501
        await self._run.connect(
            functools.partial(
                self._connect,
                encryption,
                auto_subscribe,
                rtc_config,
                single_peer_connection,
            )
        )

    async def _core_connect(self) -> None:
        # a session on the job's room connects it beside its start (`_core.SessionJob`)
        await self.connect()

    async def _connect(
        self,
        encryption: rtc.E2EEOptions | None,
        auto_subscribe: AutoSubscribe,
        rtc_config: rtc.RtcConfiguration | None,
        single_peer_connection: bool | None,
    ) -> None:
        room_options = rtc.RoomOptions(
            encryption=encryption,
            auto_subscribe=auto_subscribe == AutoSubscribe.SUBSCRIBE_ALL,
            rtc_config=rtc_config,
            single_peer_connection=single_peer_connection,
        )

        with tracer.detached_span(
            "room_connect",
            attributes={
                trace_types.ATTR_ROOM_NAME: self._info.job.room.name,
                trace_types.ATTR_ROOM_SID: self._info.job.room.sid,
                trace_types.ATTR_ROOM_AUTO_SUBSCRIBE: AutoSubscribe(auto_subscribe).value,
                trace_types.ATTR_ROOM_E2EE: encryption is not None,
            },
        ) as connect_span:
            await self._room.connect(self._info.url, self._info.token, options=room_options)
            connect_span.set_attributes(
                {
                    trace_types.ATTR_PARTICIPANT_ID: self._room.local_participant.sid,
                    trace_types.ATTR_PARTICIPANT_IDENTITY: self._room.local_participant.identity,
                    trace_types.ATTR_ROOM_REMOTE_PARTICIPANT_COUNT: len(
                        self._room.remote_participants
                    ),
                }
            )
        install_rpc_tracing(self._room.local_participant)

        # a simulator leaving ends the job; the agent under test may add other participants
        self._room.on(
            "participant_disconnected", lambda p: self._run.participant_left(dict(p.attributes))
        )
        for p in self._room.remote_participants.values():
            self._participant_available(p)

        _apply_auto_subscribe_opts(self._room, auto_subscribe)

    def delete_room(self, room_name: str | None = None) -> asyncio.Future[api.DeleteRoomResponse]:  # type: ignore
        """Deletes the room and disconnects all participants."""

        async def _delete_room() -> None:
            try:
                await self.api.room.delete_room(
                    api.DeleteRoomRequest(room=room_name or self._room.name)
                )
            except aiohttp.ServerDisconnectedError:
                logger.warning("server disconnected while deleting room")
            except api.TwirpError as e:
                if e.code != api.TwirpErrorCode.NOT_FOUND:
                    logger.warning(f"error while deleting room: {e}")
            except Exception:
                logger.exception("unknown error while deleting room")

        return cast(
            asyncio.Future[api.DeleteRoomResponse],
            self._run.api_call(_delete_room, "delete_room", api.DeleteRoomResponse()),
        )

    def add_sip_participant(
        self,
        *,
        call_to: str,
        trunk_id: str,
        participant_identity: str,
        participant_name: NotGivenOr[str] = "SIP-participant",
    ) -> asyncio.Future[api.SIPParticipantInfo]:  # type: ignore
        """
        Add a SIP participant to the room.

        Args:
            call_to: The number or SIP destination to transfer the participant to.
                         This can either be a number (+12345555555) or a
                         sip host (sip:<user>@<host>)
            trunk_id: The ID of the SIP trunk to use
            participant_identity: The identity of the participant to add
            participant_name: The name of the participant to add

        Make sure you have an outbound SIP trunk created in LiveKit.
        See https://docs.livekit.io/sip/trunk-outbound/ for more information.
        """

        def _add() -> Awaitable[api.SIPParticipantInfo]:
            return self.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=self._room.name,
                    participant_identity=participant_identity,
                    sip_trunk_id=trunk_id,
                    sip_call_to=call_to,
                    participant_name=participant_name if is_given(participant_name) else None,
                )
            )

        return cast(
            asyncio.Future[api.SIPParticipantInfo],
            self._run.api_call(_add, "add_sip_participant", api.SIPParticipantInfo()),
        )

    def transfer_sip_participant(
        self,
        participant: rtc.RemoteParticipant | str,
        transfer_to: str,
        play_dialtone: bool = False,
    ) -> asyncio.Future[TransferSIPParticipantResponse]:
        """Transfer a SIP participant to another number.

        Args:
            participant: The participant to transfer
            transfer_to: The number or SIP destination to transfer the participant to.
                         This can either be a number (+12345555555) or a
                         sip host (sip:<user>@<host>)
            play_dialtone: Whether to play a dialtone during transfer. Defaults to True.


        Returns:
            Future that completes when the transfer is complete

        Make sure you have enabled call transfer on your provider SIP trunk.
        See https://docs.livekit.io/sip/transfer-cold/ for more information.
        """

        def _transfer() -> Awaitable[TransferSIPParticipantResponse]:
            if isinstance(participant, rtc.RemoteParticipant):
                assert participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP, (
                    "Participant must be a SIP participant"
                )
                identity = participant.identity
            else:
                identity = participant
            return self.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self._room.name,
                    participant_identity=identity,
                    transfer_to=transfer_to,
                    play_dialtone=play_dialtone,
                )
            )

        return cast(
            asyncio.Future[TransferSIPParticipantResponse],
            self._run.api_call(
                _transfer, "transfer_sip_participant", TransferSIPParticipantResponse()
            ),
        )

    def shutdown(self, reason: str = "user requested") -> None:
        self._run.shutdown(reason or "user requested", True)

    def add_participant_entrypoint(
        self,
        entrypoint_fnc: Callable[[JobContext, rtc.RemoteParticipant], Coroutine[None, None, None]],
        *_: Any,
        kind: list[rtc.ParticipantKind.ValueType]
        | rtc.ParticipantKind.ValueType = DEFAULT_PARTICIPANT_KINDS,
    ) -> None:
        """Adds an entrypoint function to be run when a participant joins the room. In cases where
        the participant has already joined, the entrypoint will be run immediately. Multiple unique entrypoints can be
        added and they will each be run in parallel for each participant.
        """  # noqa: E501
        if entrypoint_fnc in self._participant_entrypoints:
            raise ValueError("entrypoints cannot be added more than once")
        self._participant_entrypoints.append(entrypoint_fnc)
        kinds = kind if isinstance(kind, list) else [kind]
        self._run.add_participant_entrypoint(
            cast(Any, entrypoint_fnc), kinds, getattr(entrypoint_fnc, "__name__", "")
        )

    def init_recording(self, options: RecordingOptions) -> None:
        redaction = self._run.init_recording(dict(options))  # type: ignore[arg-type]
        if redaction is None or (obs_url := self._run.observability_url) is None:
            self._stop_log_buffering()
            return

        logger.debug("configuring session recording")
        self._telemetry_state = _setup_cloud_tracer(
            room_id=self.job.room.sid,
            job_id=self.job.id,
            agent_name=self.job.agent_name,
            observability_url=obs_url,
            enable_traces=options["traces"],
            enable_logs=options["logs"],
            metadata=self._otel_metadata(options, redaction_enabled=redaction),
        )
        # init_recording is typically called during session.start(), at which point a bunch of
        # the logs would have already been emitted. we want to capture all of the logs as it
        # relates to the job
        self._flush_early_log_buffer(replay=options["logs"])

    def _participant_available(self, p: rtc.RemoteParticipant) -> None:
        self._run.participant_joined(self, p, p.identity, p.kind)

    def token_claims(self) -> Claims:
        return api.TokenVerifier().verify(self._info.token, verify_signature=False)

    def _otel_metadata(
        self, options: RecordingOptions | None = None, *, redaction_enabled: bool = False
    ) -> dict[str, Any] | None:
        metadata: dict[str, Any] = {}
        if (sim := self.simulation_context) is not None:
            metadata[ATTRIBUTE_SIMULATION_ENABLED] = True
            # The run/job ids ride every span and log so a run can be aggregated as a
            # whole. Omitted when blank rather than sent empty: an absent attribute
            # reads as "not a simulation" downstream, an empty one as a run named "".
            if sim.simulation_run_id:
                metadata[ATTRIBUTE_SIMULATION_RUN_ID] = sim.simulation_run_id
            if sim.simulation_job_id:
                metadata[ATTRIBUTE_SIMULATION_JOB_ID] = sim.simulation_job_id
        # stamped on every span and log so redaction can be resolved per-record, off the
        # record itself, rather than from the ambient job context. Takes the resolved
        # flag so project-wide redaction counts, not just the per-session option.
        if redaction_enabled or (options and options.get("redaction", False)):
            metadata[ATTRIBUTE_REDACTION_ENABLED] = True
        return metadata or None


def _apply_auto_subscribe_opts(room: rtc.Room, auto_subscribe: AutoSubscribe) -> None:
    if auto_subscribe not in (AutoSubscribe.AUDIO_ONLY, AutoSubscribe.VIDEO_ONLY):
        return

    def _subscribe_if_needed(pub: rtc.RemoteTrackPublication) -> None:
        if (
            auto_subscribe == AutoSubscribe.AUDIO_ONLY and pub.kind == rtc.TrackKind.KIND_AUDIO
        ) or (auto_subscribe == AutoSubscribe.VIDEO_ONLY and pub.kind == rtc.TrackKind.KIND_VIDEO):
            pub.set_subscribed(True)

    for p in room.remote_participants.values():
        for pub in p.track_publications.values():
            _subscribe_if_needed(pub)

    @room.on("track_published")
    def on_track_published(pub: rtc.RemoteTrackPublication, _: rtc.RemoteParticipant) -> None:
        _subscribe_if_needed(pub)


class JobProcess:
    def __init__(
        self,
        *,
        executor_type: JobExecutorType,
        user_arguments: Any | None,
        http_proxy: str | None,
    ) -> None:
        self._executor_type = executor_type
        self._mp_proc = mp.current_process()
        self._user_arguments = user_arguments
        self._http_proxy: str | None = http_proxy

    @property
    def executor_type(self) -> JobExecutorType:
        return self._executor_type

    @property
    def pid(self) -> int | None:
        return self._mp_proc.pid

    @property
    def user_arguments(self) -> Any | None:
        return self._user_arguments

    @property
    def http_proxy(self) -> str | None:
        return self._http_proxy


@dataclass
class SessionReport:
    job_id: str
    room_id: str
    room: str
    options: dict[str, Any] | None
    """The resolved session options (``session.options``)."""
    events: list[AgentEvent]
    chat_history: ChatContext
    audio_recording_path: Path | None = None
    audio_recording_started_at: float | None = None
    """Timestamp when the audio recording started"""
    duration: float | None = None
    started_at: float | None = None
    """Timestamp when the session started"""
    timestamp: float = field(default_factory=time.time)
    """Timestamp when the session report was created, typically at the end of the session"""
    model_usage: list[ModelUsage] | None = None
    """Usage summaries for the session, one per model/provider combination"""
    sdk_version: str = field(default_factory=lambda: __version__)
    """Version of the agents SDK"""
    # Framework reports retain their session's consent; explicit reports capture it at construction.
    _capture_content: bool = field(
        default_factory=capture_content_enabled, repr=False, kw_only=True
    )

    @property
    def _recording_options(self) -> RecordingOptionsRecord:
        # Decode the mutable legacy input once per destination into its typed native view.
        return (
            RecordingOptionsRecord(**self.options.get("recording_options", {}))
            if self.options is not None
            else RecordingOptionsRecord()
        )

    async def _snapshot(
        self,
        projection: _core._core_SessionReportProjection,
        *,
        deadline: float | None = None,
        event_sink: Callable[[object], bool] | None = None,
    ) -> tuple[SessionReportSnapshot, list[ProjectedReportEvent]]:
        from .voice.events import event_dict

        loop = asyncio.get_running_loop()
        deadline = deadline if deadline is not None else loop.time() + _core.UPLOAD_TIMEOUT_SECONDS

        event_limit = projection.event_limit(len(self.events))

        def metadata() -> SessionReportSnapshot:
            if (
                attribute_bytes(
                    (self.job_id, self.room_id, self.room, self.sdk_version, self.model_usage)
                )
                > _core.SDK_RECORD_BYTES
            ):
                raise ValueError("session report summary exceeds the native input budget")
            return projection.project(
                SessionReportSnapshot(
                    job_id=self.job_id,
                    room_id=self.room_id,
                    room=self.room,
                    events=[],
                    audio_recording_path=(
                        str(self.audio_recording_path.absolute())
                        if self.audio_recording_path
                        else None
                    ),
                    audio_recording_started_at=self.audio_recording_started_at,
                    options=(
                        _serialize_session_options(self.options)
                        if projection.content and self.options is not None
                        else None
                    ),
                    chat_history=None,
                    timestamp=self.timestamp,
                    usage=list(self.model_usage) if self.model_usage else None,
                    sdk_version=self.sdk_version,
                )
            )

        async with asyncio.timeout_at(deadline):
            snapshot = await asyncio.to_thread(metadata)
        events: list[ProjectedReportEvent] = []
        omitted = len(self.events) - event_limit
        retained_bytes = 2

        def append_event(event: AgentEvent) -> None:
            nonlocal retained_bytes, omitted
            charge = attribute_bytes(event)
            if not projection.accepts_event(retained_bytes, charge):
                omitted += 1
                return
            projected = projection.project_event(event_dict(event, json=True), charge)
            if projected is None or not projection.accepts_event(
                retained_bytes, projected.retained_bytes
            ):
                omitted += 1
                return
            if event_sink is not None:
                if not event_sink(projected.value):
                    omitted += 1
                    return
            else:
                events.append(projected)
            retained_bytes += projected.retained_bytes

        async with asyncio.timeout_at(deadline):
            for event in self.events[:event_limit]:
                await asyncio.to_thread(append_event, event)
        return replace(snapshot, omitted_events=omitted), events

    def to_dict(self) -> dict[str, Any]:
        from .voice.events import event_dict

        events_dict = [event_dict(event) for event in self.events]

        return {
            "job_id": self.job_id,
            "room_id": self.room_id,
            "room": self.room,
            "events": events_dict,
            "audio_recording_path": (
                str(self.audio_recording_path.absolute()) if self.audio_recording_path else None
            ),
            "audio_recording_started_at": self.audio_recording_started_at,
            "options": self.options,
            "chat_history": self.chat_history.to_dict(exclude_timestamp=False),
            "timestamp": self.timestamp,
            "usage": self._usage_to_dict() if self.model_usage else None,
            "sdk_version": self.sdk_version,
        }

    def _usage_to_dict(self) -> list[dict[str, Any]] | None:
        if self.model_usage is None:
            return None
        return [summary.model_dump(exclude_defaults=True) for summary in self.model_usage]
