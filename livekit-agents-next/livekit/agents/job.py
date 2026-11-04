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
import json
import logging
import multiprocessing as mp
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence, Set
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import aiohttp
from pydantic import JsonValue

import livekit.rtc as rtc
from livekit.agents._generated.job_records import (
    AutoSubscribe as AutoSubscribe,
    JobAcceptArguments as JobAcceptArguments,
    JobExecutorType as JobExecutorType,
    RunningJobInfo as RunningJobInfo,
)
from livekit.protocol import agent
from livekit.protocol.sip import TransferSIPParticipantResponse

from . import _core, telemetry
from ._generated import semconv
from ._generated.telemetry import ReportOutcome, SessionReport as SessionReport
from ._generated.types import ATTRIBUTE_SIMULATOR_DISPATCH
from .log import logger
from .types import NotGivenOr
from .utils import http_context, is_given
from .utils.participant import wait_for_participant
from .version import __version__

_JobContextVar = contextvars.ContextVar["JobContext"]("agents_job_context")


if TYPE_CHECKING:
    from livekit import api
    from livekit.api.access_token import Claims

    from .simulation import SimulationContext
    from .storage.resolve import Thread
    from .voice.agent_session import RecordingOptions, VoiceSession


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
        # every session started for the job, which the job closes; the first is the primary one,
        # which it reports
        self._sessions: list[VoiceSession] = []
        self._room.on("participant_connected", self._participant_available)

        self._log_fields: dict[str, JsonValue] = {}
        self._log_filter = _ContextLogFieldsFilter(self)
        self._handlers_with_filter: list[logging.Handler] = []

        self._thread: Thread | None = None

        # on_simulation_end callback, injected by the job runner from AgentServer.
        self._simulation_end_fnc: Callable[[SimulationContext], Any] | None = None

        self._tempdir = tempfile.TemporaryDirectory()

        self._session_directory = Path(self._tempdir.name)

        self._tagger = _core.Tagger()
        # the job's session as it closed, which every report of the job reads
        self._session_report: SessionReport | None = None

    def _on_setup(self) -> None:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(self._log_filter)
            self._handlers_with_filter.append(handler)

    async def _snapshot_session(self) -> None:
        """Once the job's session closed: finish its recording, then snapshot it. Every report of
        the job reads this snapshot; ``on_session_end`` runs after it."""
        session = self._primary_agent_session
        if session is None or self._session_report is not None:
            return
        try:
            await session._owner.finish_recording()
        except Exception:
            logger.exception("failed to finish the session recording")
        try:
            # the session's part is the core's snapshot as its loop closed
            core = await session._running.report() if session._core is not None else None
            self._session_report = await asyncio.to_thread(
                self._build_session_report, session, core, session._report_event_snapshots()
            )
        except Exception:
            logger.exception("failed to snapshot the session report")
            return

        from .voice.remote_session import ConsoleCall

        if ConsoleCall.record:
            path = Path(
                "console-recordings",
                f"session-{datetime.datetime.now().strftime('%m-%d-%H%M%S')}",
                "session_report.json",
            )
            try:
                await asyncio.to_thread(_write_console_report, self.make_session_report(), path)
            except Exception:
                logger.exception("failed to save session report")

    async def _on_cleanup(self) -> None:
        await asyncio.to_thread(self._tempdir.cleanup)

        for handler in self._handlers_with_filter:
            handler.removeFilter(self._log_filter)
        self._handlers_with_filter.clear()

    def is_fake_job(self) -> bool:
        return self._info.fake_job

    @property
    def _primary_agent_session(self) -> VoiceSession | None:
        return self._sessions[0] if self._sessions else None

    def _claim_session(self, session: VoiceSession) -> bool:
        """A session started for the job: whether it is the job's primary one."""
        if session not in self._sessions:
            self._sessions.append(session)
        return self._sessions[0] is session

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
        """The session's report: its snapshot once the job's session closed, else its state now.
        Tags, evaluations and the outcome are the tagger's as of this call."""
        session = self._report_session(session)
        report = self._session_report
        if report is None or session is not self._primary_agent_session:
            report = self._build_session_report(session, None, session._recorded_events)
        outcome = self._tagger.outcome
        return replace(
            report,
            tags=self._tagger.entries(),
            evaluations=self._tagger.evaluations,
            outcome=(
                ReportOutcome(outcome=outcome, reason=self._tagger.outcome_reason)
                if outcome is not None
                else None
            ),
        )

    def _build_session_report(
        self, session: VoiceSession, core: SessionReport | None, events: list[dict[str, Any]]
    ) -> SessionReport:
        """``core``: the session's history, item turns and usage from the core's snapshot; the
        session's current state without one (a report made while it runs)."""

        recording_started_at = session._recorder.started_at if session._recorder else None
        timestamp = time.time()
        options = {**session.options, "recording_options": dict(session._recording_options)}
        return SessionReport(
            job_id=self.job.id,
            room_id=self.job.room.sid,
            room=self.job.room.name,
            agent_name=self.job.agent_name,
            options=_report_option(options),
            events=[*events],
            chat_history=core.chat_history if core is not None else list(session.history.items),
            turns=core.turns if core is not None else {},
            usage=core.usage if core is not None else session.usage,
            started_at=session._started_at,
            timestamp=timestamp,
            duration=(
                timestamp - recording_started_at if recording_started_at is not None else None
            ),
            audio_recording_path=(
                str(session._recording_path.absolute()) if session._recording_path else None
            ),
            audio_recording_started_at=recording_started_at,
            sdk_version=__version__,
        )

    @functools.cached_property
    def api(self) -> api.LiveKitAPI:
        """Returns an LiveKitAPI for making API calls to LiveKit, with the server's URL and key
        pair: the values passed to `AgentServer`, else LIVEKIT_URL, LIVEKIT_API_KEY and
        LIVEKIT_API_SECRET.
        """
        from livekit import api

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
    def log_context_fields(self) -> dict[str, JsonValue]:
        """
        Returns the current dictionary of log fields that will be injected into log records.

        These fields enable enriched structured logging and can include job metadata,
        worker ID, trace IDs, or other diagnostic context.

        The returned dictionary can be directly edited, or entirely replaced via assignment
        (e.g., `job_context.log_context_fields = {...}`)
        """
        return self._log_fields

    @log_context_fields.setter
    def log_context_fields(self, fields: dict[str, JsonValue]) -> None:
        """
        Sets the log fields to be injected into future log records.

        Args:
            fields (dict[str, JsonValue]): A dictionary of key-value pairs representing
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

        # under the span current here: the session's start, else the job's entrypoint
        with telemetry._detached_span(
            semconv.SPAN_WAIT_FOR_PARTICIPANT,
            {semconv.LK_ROOM_IO_PARTICIPANT_FILTER: identity is not None},
        ) as span:
            participant = await wait_for_participant(self._room, identity=identity, kind=kind)
            span.set_attributes(
                {
                    semconv.LK_PARTICIPANT_ID: participant.sid,
                    semconv.LK_PII_PARTICIPANT_IDENTITY: participant.identity,
                    semconv.LK_PARTICIPANT_KIND: rtc.ParticipantKind.Name(participant.kind),
                }
            )
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

        # the room's name and id are the job identity the exporter stamps on every span
        with telemetry._detached_span(
            semconv.SPAN_ROOM_CONNECT,
            {
                semconv.LK_ROOM_AUTO_SUBSCRIBE: AutoSubscribe(auto_subscribe).value,
                semconv.LK_ROOM_E2EE: encryption is not None,
            },
        ) as connect_span:
            await self._room.connect(self._info.url, self._info.token, options=room_options)
            connect_span.set_attributes(
                {
                    semconv.LK_PARTICIPANT_ID: self._room.local_participant.sid,
                    semconv.LK_PII_PARTICIPANT_IDENTITY: self._room.local_participant.identity,
                    semconv.LK_ROOM_REMOTE_PARTICIPANT_COUNT: len(self._room.remote_participants),
                }
            )
        telemetry._trace_rpcs(self._room.local_participant)

        # a simulator leaving ends the job; the agent under test may add other participants
        self._room.on(
            "participant_disconnected", lambda p: self._run.participant_left(dict(p.attributes))
        )
        for p in self._room.remote_participants.values():
            self._participant_available(p)

        _apply_auto_subscribe_opts(self._room, auto_subscribe)

    def delete_room(self, room_name: str | None = None) -> asyncio.Future[api.DeleteRoomResponse]:  # type: ignore
        """Deletes the room and disconnects all participants."""
        from livekit import api

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
        from livekit import api

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
        from livekit import api

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
        *,
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
        self._run.init_recording(dict(options))  # type: ignore[arg-type]

    def _participant_available(self, p: rtc.RemoteParticipant) -> None:
        self._run.participant_joined(self, p, p.identity, p.kind)

    def token_claims(self) -> Claims:
        from livekit import api

        return api.TokenVerifier().verify(self._info.token, verify_signature=False)


def _report_option(value: object, depth: int = 0) -> JsonValue:
    """A session option as the report records it: an object by its class name, with the options
    its ``describe_options()`` returns (a turn detector, a model). Secrets stay out of it."""
    if depth > _core.TELEMETRY_MAX_DEPTH:
        raise ValueError("session report options are nested too deep")
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("session report option keys must be strings")
        return {key: _report_option(item, depth + 1) for key, item in value.items()}
    if isinstance(value, Sequence | Set) and not isinstance(value, bytes):
        items = [_report_option(item, depth + 1) for item in value]
        return sorted(items, key=str) if isinstance(value, Set) else items
    cls = type(value)
    name = f"{cls.__module__}.{cls.__name__}"
    if (describe := getattr(value, "describe_options", None)) is None:
        return name
    options = _report_option(describe(), depth + 1)
    if not isinstance(options, dict):
        raise TypeError("describe_options() must return a mapping")
    parts = [
        f"{key}={item if isinstance(item, str | bool | int | float) else json.dumps(item, sort_keys=True)}"
        for key, item in options.items()
        if item is not None
    ]
    return f"{name}({', '.join(parts)})"


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


def _write_console_report(report: SessionReport, path: Path) -> None:
    """The console's ``session_report.json``."""
    import json

    data = {
        **{
            key: getattr(report, key)
            for key in ("job_id", "room_id", "room", "events", "options", "timestamp")
        },
        "audio_recording_path": report.audio_recording_path,
        "audio_recording_started_at": report.audio_recording_started_at,
        "chat_history": _core.chat_ctx_to_dict(report.chat_history, exclude_timestamp=False),
        "usage": report.usage.model_dump(exclude_none=True),
        "sdk_version": report.sdk_version,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, default=str))
