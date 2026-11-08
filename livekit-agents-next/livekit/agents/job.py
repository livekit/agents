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
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import aiohttp
from pydantic import JsonValue

import livekit.rtc as rtc
from livekit.protocol import agent
from livekit.protocol.sip import TransferSIPParticipantResponse

from . import _core, telemetry
from ._generated import semconv
from ._generated.public_job import *  # noqa: F403
from ._generated.room_io import DEFAULT_PARTICIPANT_KINDS
from ._generated.telemetry import ReportJob, SessionReport as SessionReport
from ._generated.types import ATTRIBUTE_SIMULATOR_DISPATCH
from .http import client as http_client
from .log import logger
from .types import NotGivenOr
from .utils import is_given, wait_for_participant
from .version import __version__

_JobContextVar = contextvars.ContextVar["JobContext"]("agents_job_context")


if TYPE_CHECKING:
    from livekit import api
    from livekit.api.access_token import Claims

    from .evals import SimulationContext
    from .storage import Thread
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
    # In a job, options given to AgentServer in code are visible only through _server_env.
    ctx = get_job_context(required=False)
    if ctx is not None:
        return ctx._server_env.get(name)
    return os.environ.get(name)


def _server_credentials(
    url: NotGivenOr[str | None] = None,
    api_key: NotGivenOr[str | None] = None,
    api_secret: NotGivenOr[str | None] = None,
) -> tuple[str, str, str]:
    return (
        url or _server_setting("LIVEKIT_URL") or "",
        api_key or _server_setting("LIVEKIT_API_KEY") or "",
        api_secret or _server_setting("LIVEKIT_API_SECRET") or "",
    )


def current_simulation() -> SimulationContext | None:
    """The :class:`SimulationContext` of the job running on this task, or ``None``.

    ``None`` outside a simulation, including outside any job (console mode, tests).
    """
    ctx = get_job_context(required=False)
    if ctx is None:
        return None

    return ctx.simulation_context


class _ContextLogFieldsFilter(logging.Filter):
    def __init__(self, job_ctx: JobContext) -> None:
        super().__init__()
        self.job_ctx = job_ctx

    def filter(self, record: logging.LogRecord) -> bool:
        # Threaded executors share the root handlers across jobs: tag only this job's records.
        if self.job_ctx.proc.executor_type != JobExecutorType.PROCESS:
            try:
                ctx = get_job_context()
            except RuntimeError:
                return True
            else:
                if ctx != self.job_ctx:
                    return True

        for key, value in self.job_ctx._log_fields.items():
            if not hasattr(record, key):
                setattr(record, key, value)

        return True


def callable_name(callable: Any) -> str:
    for attr in ("__qualname__", "__name__"):
        if isinstance(name := getattr(callable, attr, None), str):
            return name
    return repr(callable)


class JobContext:
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
        # A job thread reads options given to AgentServer in code only from here.
        self._server_env = server_env
        self._info = info
        self._room = room
        self._run = run
        # The first session is the primary one, which the job reports.
        self._sessions: list[VoiceSession] = []

        self._log_fields: dict[str, JsonValue] = {}
        self._log_filter = _ContextLogFieldsFilter(self)
        self._handlers_with_filter: list[logging.Handler] = []

        self._thread: Thread | None = None

        self._simulation_end_fnc: Callable[[SimulationContext], Any] | None = None

        self._tempdir = tempfile.TemporaryDirectory()

        self._session_directory = Path(self._tempdir.name)

        self._tagger = _core.Tagger()
        self._session_report: SessionReport | None = None

    def _on_setup(self) -> None:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(self._log_filter)
            self._handlers_with_filter.append(handler)

    async def _snapshot_session(self) -> None:
        session = self._primary_agent_session
        if session is None or self._session_report is not None:
            return
        try:
            await session._native.finish_recording()
        except Exception:
            logger.exception("failed to finish the session recording")
        try:
            self._session_report = await session._native.final_report(
                self._report_job, session._adapter._legs(), session._report_event_snapshots()
            )
        except Exception:
            logger.exception("failed to snapshot the session report")
            return

        from .voice._session_bindings import ConsoleCall

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

    def _core_claim_primary(self, session: VoiceSession) -> bool:
        if session not in self._sessions:
            self._sessions.append(session)
        return self._sessions[0] is session

    @property
    def session_directory(self) -> Path:
        return Path(self._session_directory)

    @property
    def tagger(self) -> _core.Tagger:
        """The Tagger for adding tags and outcomes to the session, uploaded at session end.

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
        if not session._native.allows_report(self.job.id):
            raise PermissionError("A session report is available only to its owning job")
        return session

    def make_session_report(self, session: VoiceSession | None = None) -> SessionReport:
        """The session's report: its final snapshot once closed, else its state now.

        Tags, evaluations and the outcome are the tagger's as of this call.
        """
        session = self._report_session(session)
        report = self._session_report
        if report is None or session is not self._primary_agent_session:
            report = session._native.report(
                self._report_job,
                session._adapter._legs(),
                session._report_event_snapshots(),
                list(session.history.items),
            )
        return self._tagger._core_stamp(report)

    @functools.cached_property
    def _report_job(self) -> ReportJob:
        return ReportJob(
            job_id=self.job.id,
            room_id=self.job.room.sid,
            room=self.job.room.name,
            agent_name=self.job.agent_name,
            sdk_version=__version__,
        )

    @functools.cached_property
    def api(self) -> api.LiveKitAPI:
        """A LiveKitAPI client using the server's URL and credentials.

        Those are the values passed to `AgentServer`, else LIVEKIT_URL, LIVEKIT_API_KEY and
        LIVEKIT_API_SECRET.
        """
        from livekit import api

        env = self._server_env
        return api.LiveKitAPI(
            env.get("LIVEKIT_URL"),
            env.get("LIVEKIT_API_KEY"),
            env.get("LIVEKIT_API_SECRET"),
            # livekit-api only accepts an aiohttp session.
            session=http_client.http_session()._pool.client(),
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
        """The call's durable thread, resolved before the entrypoint runs.

        The dispatch's ``thread_id``, else a new thread; ``None`` on a server without storage.
        Pass it to ``VoiceSession(thread=)``.
        """
        return self._thread

    @property
    def room(self) -> rtc.Room:
        """The job's room.

        Not connected when the entrypoint starts; some properties are unavailable until
        ``connect()``.
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
        """The :class:`SimulationContext` when this job runs under a simulation, else ``None``.

        Final as soon as the entrypoint runs.
        """
        metadata = self._info.job.attributes.get(ATTRIBUTE_SIMULATOR_DISPATCH, "")
        if not metadata:
            return None

        from google.protobuf import json_format

        from livekit.protocol import agent_simulation as sim_pb

        from .evals import SimulationContext

        try:
            # Dispatches from newer servers may carry unknown fields.
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
        """Extra headers this job puts on every LiveKit Inference request.

        They override per-model headers. Empty for an ordinary job.
        """
        from .inference import HEADER_INFERENCE_PRIORITY

        headers: dict[str, str] = {}

        # Text simulations are batch load and must not compete with live traffic; audio
        # simulations stay at normal priority so their latency matches production.
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
        """Fields added to every log record of this job.

        Edit the returned dict in place, or replace it by assignment.
        """
        return self._log_fields

    @log_context_fields.setter
    def log_context_fields(self, fields: dict[str, JsonValue]) -> None:
        self._log_fields = fields

    def add_shutdown_callback(
        self,
        callback: Callable[[], Coroutine[None, None, None]]
        | Callable[[str], Coroutine[None, None, None]],
    ) -> None:
        """Add a callback run when the job shuts down; it may take the shutdown reason."""
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
        """Wait for a participant with ``identity``, or the first to join when ``None``.

        Connects to the room first if needed; returns at once if the participant is present.
        """
        if not self._room.isconnected():
            await self.connect()

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
        """Connect to the room. Call at most once.

        Args:
            encryption: End-to-end encryption options; clients must handle E2EE too.
            auto_subscribe: Which tracks to subscribe to automatically.
            rtc_config: Custom RTC configuration for the connection.
            single_peer_connection: One peer connection for publish and subscribe
                (default False).
        """
        await self._run.connect(
            functools.partial(
                self._connect,
                encryption,
                auto_subscribe,
                rtc_config,
                single_peer_connection,
            )
        )

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
        self._room.local_participant.add_rpc_interceptor(_core._core_RpcTracing(self._run))

        # A simulator leaving ends the job; other participants leaving do not.
        self._room.on(
            "participant_disconnected", lambda p: self._run.participant_left(dict(p.attributes))
        )
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
        """Dial out and add a SIP participant to the room.

        Requires an outbound SIP trunk; see https://docs.livekit.io/sip/trunk-outbound/.

        Args:
            call_to: The number (+12345555555) or SIP URI (sip:<user>@<host>) to call.
            trunk_id: The ID of the SIP trunk to use.
            participant_identity: The identity of the participant to add.
            participant_name: The name of the participant to add.
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

        Requires call transfer enabled on the provider's SIP trunk; see
        https://docs.livekit.io/sip/transfer-cold/.

        Args:
            participant: The participant to transfer, or its identity.
            transfer_to: The number (+12345555555) or SIP URI (sip:<user>@<host>).
            play_dialtone: Whether to play a dialtone during the transfer.

        Returns:
            A future that completes when the transfer is complete.
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
        self._run.shutdown(reason or "user requested")

    def init_recording(self, options: RecordingOptions) -> None:
        self._run.init_recording(dict(options))  # type: ignore[arg-type]

    def token_claims(self) -> Claims:
        from livekit import api

        return api.TokenVerifier().verify(self._info.token, verify_signature=False)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_core.session_report_json(report))
