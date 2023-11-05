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

import logging
import uuid
from dataclasses import dataclass
import asyncio
from typing import AsyncGenerator, Coroutine, Dict, Optional, Callable, Awaitable
from livekit import api, rtc
from ._proto import livekit_agent_pb2 as proto_agent
from ._proto import livekit_models_pb2 as proto_models
from urllib.parse import urlparse
import websockets

MAX_RECONNECT_ATTEMPTS = 5
ASSIGNMENT_TIMEOUT = 15


class AssignmentTimeoutError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class JobCancelledError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Worker:
    def __init__(
        self,
        available_cb: Callable[["Job"], Coroutine],
        worker_type: proto_agent.JobType.ValueType,
        ws_url: str,
        api_key: str,
        api_secret: str,
    ) -> None:
        parse_res = urlparse(ws_url)
        scheme = parse_res.scheme
        if scheme.startswith("http"):
            scheme = scheme.replace("http", "ws")

        path = parse_res.path.rstrip("/")
        url = f"{scheme}://{parse_res.netloc}/{path}"

        self._available_cb = available_cb
        self._agent_url = url + "/agent"
        self._rtc_url = url + "/rtc"
        self._wid = "AG-" + str(uuid.uuid4())[:12]
        self._worker_type = worker_type
        self._api_key = api_key
        self._api_secret = api_secret
        self._connected = False
        self._pending_jobs: Dict[str,
                                 asyncio.Future[proto_agent.JobAssignment]] = {}

    async def _connect(self) -> proto_agent.RegisterWorkerResponse:
        if self._connected:
            raise Exception("already connected")

        self._connected = True
        join_jwt = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_grants(api.VideoGrants(agent=True))
            .to_jwt()
        )

        register_req = proto_agent.WorkerMessage()
        register_req.register.worker_id = self._wid
        register_req.register.type = self._worker_type

        headers = {"Authorization": f"Bearer {join_jwt}"}
        self._ws = await websockets.connect(self._agent_url, extra_headers=headers)
        await self._ws.send(register_req.SerializeToString())

        res = await self._recv()
        return res.register

    async def _send_availability(self, job_id: str, available: bool) -> proto_agent.JobAssignment:
        """
        Send availability to the server, and wait for assignment
        """
        req = proto_agent.WorkerMessage()
        req.availability.available = available
        req.availability.job_id = job_id

        f = asyncio.Future()
        self._pending_jobs[job_id] = f
        await self._ws.send(req.SerializeToString())

        try:
            return await asyncio.wait_for(f, ASSIGNMENT_TIMEOUT)
        except asyncio.TimeoutError:
            raise AssignmentTimeoutError(
                f"assignment timeout for job {job_id}")

    async def _send_job_status(self, job_id: str, status: proto_agent.JobStatus.ValueType, error: str, user_data: str = "") -> None:
        req = proto_agent.WorkerMessage()
        req.job_update.job_id = job_id
        req.job_update.status = status
        req.job_update.error = error
        req.job_update.user_data = user_data
        await self._ws.send(req.SerializeToString())

    async def _recv(self) -> proto_agent.ServerMessage:
        message = await self._ws.recv()
        msg = proto_agent.ServerMessage()
        msg.ParseFromString(bytes(message))
        return msg

    async def _handle_new_job(self, job: "Job") -> None:
        """
        Execute the available callback, and automatically deny the job if the callback
        does not send an answer or raises an exception
        """

        try:
            await self._available_cb(job)
        except Exception as e:
            logging.error(f"available callback failed: {e}")
            return

        if not job._answered:
            logging.warn(
                f"user did not answer availability for job {job.id}, rejecting")
            await job.reject()

    async def _message_received(self, msg: proto_agent.ServerMessage) -> None:
        which = msg.WhichOneof("message")
        if which == "availability":
            # server is asking the worker if we are available for a job
            availability = msg.availability
            job = Job(self, availability.job)
            asyncio.ensure_future(self._handle_new_job(job))
        elif which == "assignment":
            # server is assigning a job to the worker
            assignment = msg.assignment
            job_id = assignment.job.id
            f = self._pending_jobs.get(job_id)
            if f is None:
                logging.error(
                    f"received assignment for unknown job {job_id}")
                return

            f.set_result(assignment)

    async def run(self):
        reg = await self._connect()
        logging.info(f"worker successfully registered: {reg}")

        try:
            while True:
                msg = await self._recv()
                await self._message_received(msg)
        except websockets.exceptions.ConnectionClosed as e:
            logging.error(f"connection closed: {e}")

    async def close(self) -> None:
        if not self._connected:
            return

        self.connected = False
        await self._ws.close()


@dataclass
class JobContext:
    job: "Job"
    room: rtc.Room
    participant: Optional[rtc.RemoteParticipant]


class Job:
    def __init__(
        self,
        worker: Worker,
        job_info: proto_agent.Job,
    ) -> None:
        self._worker = worker
        self._info = job_info
        self._room = rtc.Room()
        self._answered = False
        self._lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def room(self) -> proto_models.Room:
        return self._info.room

    @property
    def participant(self) -> Optional[proto_models.ParticipantInfo]:
        if self._info.participant.sid:
            return self._info.participant
        return None

    def _assert_not_answered(self) -> None:
        if self._answered:
            raise Exception("job already answered")

    async def reject(self) -> None:
        async with self._lock:
            self._assert_not_answered()
            self._answered = True
            await self._worker._send_availability(self.id, False)

    async def accept(
        self,
        agent: Callable[[JobContext], Coroutine],
        name: str = "",
        identity: str = "",
        metadata: str = "",
    ):
        async with self._lock:
            self._assert_not_answered()
            self._answered = True

            identity = identity or "agent-" + self.id
            grants = api.VideoGrants(
                room=self.room.name,
                room_join=True,
                agent=True,
            )

            jwt = (
                api.AccessToken(self._worker._api_key,
                                self._worker._api_secret)
                .with_identity(identity)
                .with_grants(grants)
                .with_metadata(metadata)
                .with_name(name)
                .to_jwt()
            )

            # raise AssignmentTimeoutError if assignment times out
            _ = await self._worker._send_availability(self.id, True)

            try:
                options = rtc.RoomOptions(auto_subscribe=True)
                await self._room.connect(self._worker._rtc_url, jwt, options)
            except rtc.ConnectError as e:
                logging.error(
                    f"failed to connect to the room, cancelling job {self.id}: {e}")
                await self._worker._send_job_status(self.id, proto_agent.JobStatus.JS_FAILED, str(e))
                raise e

            participant = None
            if self.participant is not None:
                # cancel the job if the participant cannot be found
                # this can happen if the participant has left the room before the agent gets the job
                participant = self._room.participants.get(self.participant.sid)
                if participant is None:
                    logging.warn(
                        f"participant '{self.participant.sid}' not found, cancelling job {self.id}")
                    await self._worker._send_job_status(self.id, proto_agent.JobStatus.JS_FAILED, "participant not found")
                    await self._room.disconnect()
                    raise JobCancelledError(
                        f"participant '{self.participant.sid}' not found")

            # start the agent
            job_ctx = JobContext(job=self, room=self._room,
                                 participant=participant)
            asyncio.ensure_future(agent(job_ctx))
