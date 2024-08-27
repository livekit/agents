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
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine, Set, Awaitable, TypeVar

from livekit import rtc
from livekit.protocol import agent, models
from .log import logger

T = TypeVar('T')

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


@dataclass
class RunningJobInfo:
    accept_arguments: JobAcceptArguments
    job: agent.Job
    url: str
    token: str

_ParticipantFilterFnc = Callable[[int, rtc.RemoteParticipant], bool]




class JobContext:
    def __init__(
        self,
        *,
        proc: JobProcess,
        info: RunningJobInfo,
        room: rtc.Room,
        on_connect: Callable[[], None],
        on_shutdown: Callable[[str], None],
    ) -> None:
        self._proc = proc
        self._info = info
        self._room = room
        self._on_connect = on_connect
        self._on_shutdown = on_shutdown
        self._shutdown_callbacks: list[Callable[[], Coroutine[None, None, None]]] = []
        self._participant_coro_lookup = dict[
            Callable[[rtc.RemoteParticipant], bool],
            Callable[[rtc.RemoteParticipant], Coroutine[None, Any, None]],
        ]()
        self._participant_tasks = set[asyncio.Task[Any]]()
        self._room.on("participant_connected", self._on_participant_connected)

    @property
    def proc(self) -> JobProcess:
        return self._proc

    @property
    def job(self) -> agent.Job:
        return self._info.job

    @property
    def room(self) -> rtc.Room:
        return self._room

    @property
    def agent(self) -> rtc.LocalParticipant:
        return self._room.local_participant

    def add_shutdown_callback(
        self, callback: Callable[[], Coroutine[None, None, None]]
    ) -> None:
        self._shutdown_callbacks.append(callback)

    async def connect(
        self,
        *,
        e2ee: rtc.E2EEOptions | None = None,
        auto_subscribe: AutoSubscribe = AutoSubscribe.SUBSCRIBE_ALL,
        rtc_config: rtc.RtcConfiguration | None = None,
    ) -> None:
        room_options = rtc.RoomOptions(
            e2ee=e2ee,
            auto_subscribe=auto_subscribe == AutoSubscribe.SUBSCRIBE_ALL,
            rtc_config=rtc_config,
        )

        await self._room.connect(self._info.url, self._info.token, options=room_options)
        self._on_connect()
        for p in self._room.remote_participants.values():
            self._on_participant_connected(p)

        _apply_auto_subscribe_opts(self._room, auto_subscribe)

    def shutdown(self, reason: str = "") -> None:
        self._on_shutdown(reason)

    def _on_participant_connected(self, p: rtc.RemoteParticipant) -> None:
        _participant_coro_lookup_copy = self._participant_coro_lookup.copy()
        for filter, coro in _participant_coro_lookup_copy.items():
            if filter(p):
                task = asyncio.create_task(coro(p))
                self._participant_tasks.add(task)
                task.add_done_callback(self._participant_tasks.remove)
                self._participant_coro_lookup.pop(filter)

    def add_participant_task(
        self,
        *,
        task_fnc: Callable[[rtc.RemoteParticipant], Coroutine[None, None, T]],
        identity: str | None = None,
        filter_fnc: Callable[[rtc.RemoteParticipant], bool] | None = None,
    ) -> asyncio.Future[T]:
        """Adds a task to be run when a participant that matches the filter joins the room. In cases where
        the participant has already joined, the task will be run immediately. The task will only be run once
        per participant. If the task needs to be run again, it should be re-added after awaiting the future that
        is returned.
        """
        def _filter_fnc(p: rtc.RemoteParticipant) -> bool:
            if filter_fnc and identity:
                raise ValueError("cannot specify both identity and filter_fnc")

            if identity:
                return p.identity == identity

            if filter_fnc:
                return filter_fnc(p)

            return True

        fut = asyncio.Future[T]()
        async def _coro_wrapper(p: rtc.RemoteParticipant):
            try:
                res = await task_fnc(p)
                fut.set_result(res)
            except Exception as e:
                fut.set_exception(e)



        self._participant_coro_lookup[_filter_fnc] = _coro_wrapper
        return fut


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
    def __init__(self, *, start_arguments: Any | None = None) -> None:
        self._mp_proc = mp.current_process()
        self._userdata: dict[str, Any] = {}
        self._start_arguments = start_arguments

    @property
    def pid(self) -> int | None:
        return self._mp_proc.pid

    @property
    def userdata(self) -> dict:
        return self._userdata

    @property
    def start_arguments(self) -> Any | None:
        return self._start_arguments


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
    ) -> None:
        """Accept the job request, and start the job if the LiveKit SFU assigns the job to our worker."""
        if not identity:
            identity = "agent-" + self.id

        accept_arguments = JobAcceptArguments(
            name=name,
            identity=identity,
            metadata=metadata,
        )

        await self._on_accept(accept_arguments)
