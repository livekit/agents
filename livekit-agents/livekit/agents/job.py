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
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Callable, Coroutine, Tuple

from livekit import api, rtc
from livekit.protocol import agent, models

from .ipc.inference_executor import InferenceExecutor
from .log import logger

_JobContextVar = contextvars.ContextVar["JobContext"]("agents_job_context")


def get_current_job_context() -> JobContext:
    ctx = _JobContextVar.get(None)
    if ctx is None:
        raise RuntimeError(
            "no job context found, are you running this code inside a job entrypoint?"
        )

    return ctx


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
        self._shutdown_callbacks: list[Callable[[], Coroutine[None, None, None]]] = []
        self._participant_entrypoints: list[
            Tuple[
                Callable[
                    [JobContext, rtc.RemoteParticipant], Coroutine[None, None, None]
                ],
                list[rtc.ParticipantKind.ValueType] | rtc.ParticipantKind.ValueType,
            ]
        ] = []
        self._participant_tasks = dict[Tuple[str, Callable], asyncio.Task[None]]()
        self._room.on("participant_connected", self._participant_available)
        self._inf_executor = inference_executor

    @property
    def inference_executor(self) -> InferenceExecutor:
        return self._inf_executor

    @functools.cached_property
    def api(self) -> api.LiveKitAPI:
        return api.LiveKitAPI()

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

    def add_shutdown_callback(
        self, callback: Callable[[], Coroutine[None, None, None]]
    ) -> None:
        self._shutdown_callbacks.append(callback)

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
        if not self._room.isconnected():
            raise RuntimeError("room is not connected")

        fut = asyncio.Future[rtc.RemoteParticipant]()

        def kind_match(p: rtc.RemoteParticipant) -> bool:
            if isinstance(kind, list):
                return p.kind in kind

            return p.kind == kind

        for p in self._room.remote_participants.values():
            if (identity is None or p.identity == identity) and kind_match(p):
                fut.set_result(p)
                break

        def _on_participant_connected(p: rtc.RemoteParticipant):
            if (identity is None or p.identity == identity) and kind_match(p):
                self._room.off("participant_connected", _on_participant_connected)
                if not fut.done():
                    fut.set_result(p)

        if not fut.done():
            self._room.on("participant_connected", _on_participant_connected)

        return await fut

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
        """
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

    def shutdown(self, reason: str = "") -> None:
        self._on_shutdown(reason)

    def add_participant_entrypoint(
        self,
        entrypoint_fnc: Callable[
            [JobContext, rtc.RemoteParticipant], Coroutine[None, None, None]
        ],
        *_,
        kind: list[rtc.ParticipantKind.ValueType]
        | rtc.ParticipantKind.ValueType = DEFAULT_PARTICIPANT_KINDS,
    ):
        """Adds an entrypoint function to be run when a participant joins the room. In cases where
        the participant has already joined, the entrypoint will be run immediately. Multiple unique entrypoints can be
        added and they will each be run in parallel for each participant.
        """

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
                    f"a participant has joined before a prior participant task matching the same identity has finished: '{p.identity}'"
                )
            task_name = f"part-entry-{p.identity}-{coro.__name__}"
            task = asyncio.create_task(coro(self, p), name=task_name)
            self._participant_tasks[(p.identity, coro)] = task
            task.add_done_callback(
                lambda _: self._participant_tasks.pop((p.identity, coro))
            )


def _apply_auto_subscribe_opts(room: rtc.Room, auto_subscribe: AutoSubscribe) -> None:
    if auto_subscribe not in (AutoSubscribe.AUDIO_ONLY, AutoSubscribe.VIDEO_ONLY):
        return

    def _subscribe_if_needed(pub: rtc.RemoteTrackPublication):
        if (
            auto_subscribe == AutoSubscribe.AUDIO_ONLY
            and pub.kind == rtc.TrackKind.KIND_AUDIO
        ) or (
            auto_subscribe == AutoSubscribe.VIDEO_ONLY
            and pub.kind == rtc.TrackKind.KIND_VIDEO
        ):
            pub.set_subscribed(True)

    for p in room.remote_participants.values():
        for pub in p.track_publications.values():
            _subscribe_if_needed(pub)

    @room.on("track_published")
    def on_track_published(pub: rtc.RemoteTrackPublication, _: rtc.RemoteParticipant):
        _subscribe_if_needed(pub)


class JobProcess:
    def __init__(
        self,
        *,
        user_arguments: Any | None = None,
    ) -> None:
        self._mp_proc = mp.current_process()
        self._userdata: dict[str, Any] = {}
        self._user_arguments = user_arguments

    @property
    def pid(self) -> int | None:
        return self._mp_proc.pid

    @property
    def userdata(self) -> dict:
        return self._userdata

    @property
    def user_arguments(self) -> Any | None:
        return self._user_arguments


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
        """Accept the job request, and start the job if the LiveKit SFU assigns the job to our worker."""
        if not identity:
            identity = "agent-" + self.id

        accept_arguments = JobAcceptArguments(
            name=name,
            identity=identity,
            metadata=metadata,
            attributes=attributes,
        )

        await self._on_accept(accept_arguments)
