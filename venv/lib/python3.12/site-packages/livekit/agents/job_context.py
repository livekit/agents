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

import asyncio
import contextlib
import logging
from typing import Coroutine, Optional, TYPE_CHECKING
from livekit import api, rtc, protocol

# TODO: refactor worker so we can avoid this circular import
if TYPE_CHECKING:
    from .worker import Worker


class JobContext:
    """Context for job, it contains the worker, the room, and the participant.
    You should not create these on your own. They are created by the Worker.
    """

    def __init__(
        self,
        id: str,
        worker: "Worker",
        room: rtc.Room,
        participant: Optional[rtc.Participant] = None,
    ) -> None:
        self._id = id
        self._worker = worker
        self._room = room
        self._participant = participant
        self._closed = False
        self._lock = asyncio.Lock()
        self._worker._running_jobs.append(self)
        self._tasks = set()

    @property
    def id(self) -> str:
        """Job ID"""
        return self._id

    @property
    def room(self) -> rtc.Room:
        """LiveKit Room object"""
        return self._room

    @property
    def participant(self) -> Optional[rtc.Participant]:
        """LiveKit RemoteParticipant that the Job launched for. None if Agent is launched for the Room."""
        return self._participant

    @property
    def agent(self) -> rtc.LocalParticipant:
        """The agent's Participant identity"""
        return self._room.local_participant

    @property
    def api(self) -> api.LiveKitAPI:
        """LiveKit API client"""
        return self._worker.api

    def create_task(self, coro: Coroutine) -> asyncio.Task:
        """Schedule the execution of a coroutine object in a spawn task using asyncio and keep a reference to it.
        The task is automatically cancelled when the job is disconnected.
        """

        t = asyncio.create_task(coro)

        def done_cb(task: asyncio.Task):
            self._tasks.discard(t)
            if not task.cancelled() and task.exception():
                logging.error(
                    "A task raised an exception:",
                    exc_info=task.exception(),
                    extra=self.logging_extra,
                )

        t.add_done_callback(done_cb)
        return t

    async def disconnect(self) -> None:
        """
        Disconnect the agent from the room, shutdown the job, and cleanup resources.
        This will also cancel all tasks created by this job if task_timeout is specified.

        Args:
            task_timeout (Optional[float], optional): How long to wait before tasks created via JobContext.create_task are cancelled.
            If None, tasks will not be cancelled. Defaults to 25.
        """

        async with self._lock:
            logging.info("shutting down job %s", self.id, extra=self.logging_extra)
            if self._closed:
                return

            self._worker._running_jobs.remove(self)
            self._closed = True
            await self.room.disconnect()

            for task in self._tasks:
                task.cancel()

            for task in self._tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            logging.info("job %s shutdown", self.id, extra=self.logging_extra)

    async def update_status(
        self,
        status: protocol.agent.JobStatus.ValueType,
        error: str = "",
        user_data: str = "",
    ) -> None:
        await self._worker._send_job_status(self._id, status, error, user_data)

    @property
    def logging_extra(self) -> dict:
        """
        Additional context to identify the job in logs.

        Usage: logging.info("my message", extra=ctx.logging_extra)
        """
        e = {
            "job_id": self.id,
            "room": self.room.name,
            "agent_identity": self.agent.identity,
            "worker_id": self._worker.id,
        }
        if self.participant:
            e["participant_identity"] = self.participant.identity
        return e
