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

from livekit import rtc
from livekit.protocol import agent

from . import aio


class JobContext:
    """Context for job, it contains the worker, the room, and the participant.
    You should not create these on your own. They are created by the Worker."""

    def __init__(
        self,
        close_tx: aio.ChanSender[None],
        job: agent.Job,
        room: rtc.Room,
        publisher: rtc.RemoteParticipant | None = None,
    ) -> None:
        self._job = job
        self._room = room
        self._publisher = publisher
        self._close_tx = close_tx

    @property
    def id(self) -> str:
        return self._job.id

    @property
    def job(self) -> agent.Job:
        return self._job

    @property
    def room(self) -> rtc.Room:
        return self._room

    @property
    def publisher(self) -> rtc.RemoteParticipant | None:
        return self._publisher

    @property
    def agent(self) -> rtc.LocalParticipant:
        return self._room.local_participant

    def shutdown(self) -> None:
        self._close_tx.close()
