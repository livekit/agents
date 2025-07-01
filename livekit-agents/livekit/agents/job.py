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
import functools
import inspect
import logging
import multiprocessing as mp
from collections.abc import Coroutine
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable

import jwt

from livekit import api, rtc
from livekit.protocol import agent, models

from .ipc.inference_executor import InferenceExecutor
from .log import logger
from .types import NOT_GIVEN, NotGivenOr
from .utils import http_context, is_given, wait_for_participant

_JobContextVar = contextvars.ContextVar["JobContext"]("agents_job_context")


def get_job_context() -> JobContext:
    ctx = _JobContextVar.get(None)
    if ctx is None:
        raise RuntimeError(
            "no job context found, are you running this code inside a job entrypoint?"
        )

    return ctx


get_current_job_context = get_job_context


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


DEFAULT_PARTICIPANT_KINDS: list[rtc.ParticipantKind.ValueType] = [
    rtc.ParticipantKind.PARTICIPANT_KIND_SIP,
    rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
]


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
        on_connect: Callable[[], None],
        on_shutdown: Callable[[str], None],
        inference_executor: InferenceExecutor,
    ) -> None:
        self._proc = proc
        self._info = info
        self._room = room
        self._on_connect = on_connect
        self._on_shutdown = on_shutdown
        self._shutdown_callbacks: list[Callable[[str], Coroutine[None, None, None]]] = []
        self._tracing_callbacks: list[Callable[[], Coroutine[None, None, None]]] = []
        self._participant_entrypoints: list[
            tuple[
                JobContext._PARTICIPANT_ENTRYPOINT_CALLBACK,
                list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType,
            ]
        ] = []
        self._participant_tasks = dict[
            tuple[str, JobContext._PARTICIPANT_ENTRYPOINT_CALLBACK], asyncio.Task[None]
        ]()
        self._pending_tasks = list[asyncio.Task[Any]]()
        self._room.on("participant_connected", self._participant_available)
        self._inf_executor = inference_executor

        self._init_log_factory()
        self._log_fields: dict[str, Any] = {}

        self._connected = False
        self._lock = asyncio.Lock()

    def _init_log_factory(self) -> None:
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = old_factory(*args, **kwargs)

            if self.proc.executor_type != JobExecutorType.PROCESS:
                try:
                    ctx = get_job_context()
                except RuntimeError:
                    return record
                else:
                    if ctx != self:
                        return record

            for key, value in self._log_fields.items():
                setattr(record, key, value)

            return record

        logging.setLogRecordFactory(record_factory)

    @property
    def inference_executor(self) -> InferenceExecutor:
        return self._inf_executor

    @functools.cached_property
    def api(self) -> api.LiveKitAPI:
        """Returns an LiveKitAPI for making API calls to LiveKit.

        This property requires LIVEKIT_API_KEY and LIVEKIT_API_SECRET to be set in the environment.
        If they are passed in WorkerOptions, it would not be able to satisfy this API.
        """
        return api.LiveKitAPI(session=http_context.http_session())

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

    def add_tracing_callback(
        self,
        callback: Callable[[], Coroutine[None, None, None]],
    ) -> None:
        """
        Add a callback to be called when the job is about to receive a new tracing request.
        """
        self._tracing_callbacks.append(callback)

    def add_shutdown_callback(
        self,
        callback: Callable[[], Coroutine[None, None, None]]
        | Callable[[str], Coroutine[None, None, None]],
    ) -> None:
        """
        Add a callback to be called when the job is shutting down.
        Optionally the callback can take a single argument, the shutdown reason.
        """
        min_args_num = 2 if inspect.ismethod(callback) else 1
        if callback.__code__.co_argcount >= min_args_num:
            self._shutdown_callbacks.append(callback)  # type: ignore
        else:

            async def wrapper(_: str) -> None:
                await callback()  # type: ignore

            self._shutdown_callbacks.append(wrapper)

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
        return await wait_for_participant(self._room, identity=identity, kind=kind)

    async def connect(
        self,
        *,
        e2ee: rtc.E2EEOptions | None = None,
        auto_subscribe: AutoSubscribe = AutoSubscribe.SUBSCRIBE_ALL,
        rtc_config: rtc.RtcConfiguration | None = None,
    ) -> None:
        """Connect to the room. This method should be called only once.

        Args:
            e2ee: End-to-end encryption options. If provided, the Agent will utilize end-to-end encryption. Note: clients will also need to handle E2EE.
            auto_subscribe: Whether to automatically subscribe to tracks. Default is AutoSubscribe.SUBSCRIBE_ALL.
            rtc_config: Custom RTC configuration to use when connecting to the room.
        """  # noqa: E501
        async with self._lock:
            if self._connected:
                return

            room_options = rtc.RoomOptions(
                e2ee=e2ee,
                auto_subscribe=auto_subscribe == AutoSubscribe.SUBSCRIBE_ALL,
                rtc_config=rtc_config,
            )

            await self._room.connect(self._info.url, self._info.token, options=room_options)
            self._on_connect()
            for p in self._room.remote_participants.values():
                self._participant_available(p)

            _apply_auto_subscribe_opts(self._room, auto_subscribe)
            self._connected = True

    def delete_room(self) -> asyncio.Future[api.DeleteRoomResponse]:  # type: ignore
        """Deletes the room and disconnects all participants."""
        task = asyncio.create_task(
            self.api.room.delete_room(api.DeleteRoomRequest(room=self._room.name))
        )
        self._pending_tasks.append(task)
        task.add_done_callback(lambda _: self._pending_tasks.remove(task))
        return task

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
        task = asyncio.create_task(
            self.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=self._room.name,
                    participant_identity=participant_identity,
                    sip_trunk_id=trunk_id,
                    sip_call_to=call_to,
                    participant_name=participant_name if is_given(participant_name) else None,
                )
            ),
        )
        self._pending_tasks.append(task)
        task.add_done_callback(lambda _: self._pending_tasks.remove(task))
        return task

    def transfer_sip_participant(
        self,
        participant: rtc.RemoteParticipant | str,
        transfer_to: str,
        play_dialtone: bool = False,
    ) -> asyncio.Future[api.SIPParticipantInfo]:  # type: ignore
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
        if isinstance(participant, rtc.RemoteParticipant):
            assert participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP, (
                "Participant must be a SIP participant"
            )
            participant_identity = participant.identity
        else:
            participant_identity = participant

        task = asyncio.create_task(
            self.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=self._room.name,
                    participant_identity=participant_identity,
                    transfer_to=transfer_to,
                    play_dialtone=play_dialtone,
                )
            ),
        )
        self._pending_tasks.append(task)
        task.add_done_callback(lambda _: self._pending_tasks.remove(task))
        return task

    def shutdown(self, reason: str = "") -> None:
        self._on_shutdown(reason)

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

        if entrypoint_fnc in [e for (e, _) in self._participant_entrypoints]:
            raise ValueError("entrypoints cannot be added more than once")

        self._participant_entrypoints.append((entrypoint_fnc, kind))

    def _participant_available(self, p: rtc.RemoteParticipant) -> None:
        for coro, kind in self._participant_entrypoints:
            if isinstance(kind, list):
                if p.kind not in kind:
                    continue
            else:
                if p.kind != kind:
                    continue

            if (p.identity, coro) in self._participant_tasks:
                logger.warning(
                    f"a participant has joined before a prior participant task matching the same identity has finished: '{p.identity}'"  # noqa: E501
                )
            task_name = f"part-entry-{p.identity}-{coro.__name__}"
            task = asyncio.create_task(coro(self, p), name=task_name)
            self._participant_tasks[(p.identity, coro)] = task
            task.add_done_callback(
                lambda _, coro=coro: self._participant_tasks.pop((p.identity, coro))  # type: ignore
            )

    def decode_token(self, api_secret: NotGivenOr[str] = NOT_GIVEN) -> dict[str, Any]:
        options = {}
        if not is_given(api_secret):
            options["verify_signature"] = False
            api_secret = ""
        return jwt.decode(self._info.token, api_secret, options=options, algorithms=["HS256"])  # type: ignore


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
        self._userdata: dict[str, Any] = {}
        self._user_arguments = user_arguments
        self._http_proxy: str | None = http_proxy

    @property
    def executor_type(self) -> JobExecutorType:
        return self._executor_type

    @property
    def pid(self) -> int | None:
        return self._mp_proc.pid

    @property
    def userdata(self) -> dict[Any, Any]:
        return self._userdata

    @property
    def user_arguments(self) -> Any | None:
        return self._user_arguments

    @property
    def http_proxy(self) -> str | None:
        return self._http_proxy


class JobRequest:
    def __init__(
        self,
        *,
        job: agent.Job,
        on_reject: Callable[[], Coroutine[None, None, None]],
        on_accept: Callable[[JobAcceptArguments], Coroutine[None, None, None]],
    ) -> None:
        self._job = job
        self._lock = asyncio.Lock()
        self._on_reject = on_reject
        self._on_accept = on_accept

    @property
    def id(self) -> str:
        return self._job.id

    @property
    def job(self) -> agent.Job:
        return self._job

    @property
    def room(self) -> models.Room:
        return self._job.room

    @property
    def publisher(self) -> models.ParticipantInfo | None:
        return self._job.participant

    @property
    def agent_name(self) -> str:
        return self._job.agent_name

    async def reject(self) -> None:
        """Reject the job request. The job may be assigned to another worker"""
        await self._on_reject()

    async def accept(
        self,
        *,
        name: str = "",
        identity: str = "",
        metadata: str = "",
        attributes: dict[str, str] | None = None,
    ) -> None:
        """Accept the job request, and start the job if the LiveKit SFU assigns the job to our worker."""  # noqa: E501
        if not identity:
            identity = "agent-" + self.id

        accept_arguments = JobAcceptArguments(
            name=name,
            identity=identity,
            metadata=metadata,
            attributes=attributes,
        )

        await self._on_accept(accept_arguments)
