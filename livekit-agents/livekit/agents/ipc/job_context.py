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
import contextlib
import logging
from typing import Coroutine

from livekit import rtc
from livekit.protocol import agent, worker


class JobContext:
    """Context for job, it contains the worker, the room, and the participant.
    You should not create these on your own. They are created by the Worker.
    """

    def __init__(
        self,
        job: agent.Job,
        room: rtc.Room,
        participant: rtc.RemoteParticipant | None = None,
    ) -> None:
        self._job = job
        self._room = room
        self._participant = participant
        self._tasks = set()
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        """Job ID"""
        return self._job.id

    @property
    def room(self) -> rtc.Room:
        """LiveKit Room object"""
        return self._room

    @property
    def participant(self) -> rtc.RemoteParticipant | None:
        """LiveKit RemoteParticipant that the Job launched for. None if Agent is launched for the Room."""
        return self._participant

    @property
    def agent(self) -> rtc.LocalParticipant:
        """The agent's Participant identity"""
        return self._room.local_participant

    def create_task(self, coro: Coroutine) -> asyncio.Task:
        """Schedule the execution of a coroutine object in a spawn task and keep a reference to it.
        The task is automatically cancelled when the job is terminated.
        """

        t = asyncio.create_task(coro)

        def done_cb(task: asyncio.Task):
            self._tasks.discard(t)
            if not task.cancelled() and task.exception():
                logging.error("task raised an exception:", exc_info=task.exception())

        t.add_done_callback(done_cb)
        return t

    async def shutdown(self) -> None:
        async with self._lock:
            await self._ipc_client.send(
                worker.IPCJobMessage(job_shutdown=worker.JobShutdownRequest())
            )

            await self._room.disconnect()

            for task in self._tasks:
                task.cancel()

            for task in self._tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def update_status(
        self,
        userdata: str,
    ) -> None:
        await self._ipc_client.send(
            worker.IPCJobMessage(
                update_job=worker.IPCUpdateJobStatus(userdata=userdata)
            )
        )
