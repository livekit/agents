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
from typing import Any, Callable, Coroutine

from livekit import rtc
from livekit.protocol import agent, models


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

        _apply_auto_subscribe_opts(self._room, auto_subscribe)

    def shutdown(self, reason: str = "") -> None:
        self._on_shutdown(reason)


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
