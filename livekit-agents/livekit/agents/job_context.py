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
import logging
from typing import Coroutine, Optional, TYPE_CHECKING
from livekit import rtc, protocol

# TODO: refactor worker so we can avoid this circular import
if TYPE_CHECKING:
    from .worker import Worker


class JobContext:
    """
    Context for job, it contains the worker, the room, and the participant.
    You should not create these on your own. They are created by the Worker.
    """

    def __init__(
        self,
        id: str,
        worker: "Worker",
        room: rtc.Room,
        agent_identity: str,
        participant: Optional[rtc.Participant] = None,
    ) -> None:
        self._id = id
        self._worker = worker
        self._room = room
        self._agent_identity = agent_identity
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
        """LiveKit Participant corresponding to the Job"""
        return self._participant

    @property
    def agent_identity(self) -> Optional[rtc.Participant]:
        """Participant sid for the agent"""
        return self._agent_identity

    def create_task(self, coro: Coroutine) -> asyncio.Task:
        """
        Creates and asyncio.Task and internally
        keeps a reference until the task is complete to prevent
        it from being garbage collected. Also adds a callback to
        that logs the task's completion status along with any
        exceptions that may have been raised.

        Args:
            coro (Coroutine): async function to run

        Returns:
            asyncio.Task
        """

        t = asyncio.create_task(coro)

        def done_cb(task: asyncio.Task):
            self._tasks.discard(t)
            if task.exception():
                logging.error("A task raised an exception:", exc_info=task.exception())

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
            logging.info("shutting down job %s", self.id)
            if self._closed:
                return

            self._worker._running_jobs.remove(self)
            self._closed = True
            await self.room.disconnect()

            for task in self._tasks:
                task.cancel()

            logging.info("job %s shutdown", self.id)

    async def update_status(
        self,
        status: protocol.agent.JobStatus.ValueType,
        error: str = "",
        user_data: str = "",
    ) -> None:
        await self._worker._send_job_status(self._id, status, error, user_data)
