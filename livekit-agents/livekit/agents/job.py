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
from typing import Any, Callable, Optional, Union

from livekit import rtc
from livekit.protocol import agent, models

from . import utils
from .exceptions import AvailabilityAnsweredError
from .log import logger


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
    job: agent.Job
    accept_args: JobAcceptArguments
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

        if auto_subscribe in (AutoSubscribe.AUDIO_ONLY, AutoSubscribe.VIDEO_ONLY):
            pass

    def shutdown(self, reason: str = "") -> None:
        self._on_shutdown(reason)


class JobProcess:
    def __init__(self, *, start_arguments: Any | None = None) -> None:
        self._mp_proc = mp.current_process()
        self._userdata = {}
        self._start_arguments = start_arguments

    @property
    def pid(self) -> int:
        return self._mp_proc.pid

    @property
    def userdata(self) -> dict:
        return self._userdata

    @property
    def start_arguments(self) -> Any | None:
        return self._start_arguments


@dataclass
class _AvailRes:
    avail: bool
    data: _JobAcceptArgs | None = None
    assignment_tx: utils.aio.ChanSender[Optional[BaseException]] | None = None


class JobRequest:
    def __init__(
        self, job: agent.Job, answer_tx: utils.aio.ChanSender[_AvailRes]
    ) -> None:
        self._job = job
        self._lock = asyncio.Lock()
        self._answer_tx = answer_tx
        self._answered = False

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
    def answered(self) -> bool:
        return self._answered

    async def reject(self) -> None:
        async with self._lock:
            if self._answered:
                raise AvailabilityAnsweredError
            self._answered = True

        await self._answer_tx.send(_AvailRes(avail=False))
        logger.info(f"rejected job {self.id}", extra={"job": self.job})

    async def accept(
        self,
        *,
        name: str = "",
        identity: str = "",
        metadata: str = "",
    ) -> None:
        async with self._lock:
            if self._answered:
                raise AvailabilityAnsweredError
            self._answered = True

        if not identity:
            identity = "agent-" + self.id

        assign_tx = assign_rx = utils.aio.Chan[Union[BaseException, None]](1)
        data = _JobAcceptArgs(
            name=name,
            identity=identity,
            metadata=metadata,
        )
        await self._answer_tx.send(
            _AvailRes(avail=True, data=data, assignment_tx=assign_tx)
        )

        # wait for the server to accept the assignment
        # this will raise a TimeoutError if the server does not respond
        # or if an exception is raised
        exc = await assign_rx.recv()

        if exc is not None:
            raise exc

        logger.info(f"accepted job {self.id}", extra={"job": self.job})
