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
from enum import Enum
from typing import Callable, Coroutine

from attr import define
from livekit.protocol import agent, models

from . import aio
from .exceptions import AvailabilityAnsweredError
from .job_context import JobContext
from .log import logger

AutoDisconnect = Enum("AutoDisconnect", ["ROOM_EMPTY", "PUBLISHER_LEFT", "NONE"])
AutoSubscribe = Enum(
    "AutoSubscribe", ["SUBSCRIBE_ALL", "SUBSCRIBE_NONE", "VIDEO_ONLY", "AUDIO_ONLY"]
)

AgentEntry = Callable[[JobContext], Coroutine]


@define(kw_only=True)
class AcceptData:
    entry: AgentEntry
    auto_subscribe: AutoSubscribe
    auto_disconnect: AutoDisconnect
    name: str
    identity: str
    metadata: str


@define(kw_only=True)
class AvailRes:
    avail: bool
    data: AcceptData | None = None
    assignment_tx: aio.ChanSender[BaseException | None] | None = None


class JobRequest:
    def __init__(self, job: agent.Job, answer_tx: aio.ChanSender[AvailRes]) -> None:
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

        await self._answer_tx.send(AvailRes(avail=False))
        logger.info(f"rejected job {self.id}", extra={"job": self.job})

    async def accept(
        self,
        entry: AgentEntry,
        *,
        auto_subscribe: AutoSubscribe = AutoSubscribe.SUBSCRIBE_ALL,
        auto_disconnect: AutoDisconnect = AutoDisconnect.ROOM_EMPTY,
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

        assign_tx, assign_rx = aio.channel(1)
        data = AcceptData(
            entry=entry,
            auto_subscribe=auto_subscribe,
            auto_disconnect=auto_disconnect,
            name=name,
            identity=identity,
            metadata=metadata,
        )
        await self._answer_tx.send(
            AvailRes(avail=True, data=data, assignment_tx=assign_tx)
        )

        # wait for the server to accept the assignment
        # this will raise a TimeoutError if the server does not respond
        # or if an exception is raised
        exc = await assign_rx.recv()

        if exc is not None:
            raise exc

        logger.info(f"accepted job {self.id}", extra={"job": self.job})
