import logging
import uuid
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable
from .job import Job

import livekit.api as api
import livekit.agents._proto.livekit_agent_pb2 as proto
import livekit.agents._proto.livekit_models_pb2 as model_proto


class Worker(ABC):

    @dataclass
    class Handler:
        agent_identity_generator: Optional[Callable[[str], str]]
        job_available_cb: Callable[["Worker", Job], Awaitable[None]]

    def __init__(self, *,
                 ws_url: str,
                 api_key: str,
                 api_secret: str,
                 handler: Handler):
        self._active_jobs: [Job] = []
        self._ws_url = ws_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._handler = handler

    async def simulate_job(self, *, job_type: proto.JobType, room: str, participant_sid: Optional[str] = None):
        job_id = f"simulated-{uuid.uuid4()}"
        job = proto.Job(id=job_id, type=type, room=room)

        if job_type == proto.JobType.JT_ROOM:
            if participant_sid is not None:
                raise ValueError("participant_sid must be None for JT_ROOM")
        elif job_type == proto.JobType.JT_PARTICIPANT:
            if participant_sid is None:
                raise ValueError(
                    "participant_sid must be provided for JT_PARTICIPANT")
            job.participant = model_proto.ParticipantInfo()

        await self._on_job_available(job)

    async def _on_job_available(self, job: proto.Job):
        async def worker_accept_cb():
            await self._accept_job(job)
            self._active_jobs.append(job)

        job = Job(job_id=job.id,
                  ws_url=self._ws_url,
                  token=self.generate_token(
                      identity=job.id, room=job.room),
                  room=job.room,
                  participant_sid=job.participant.sid,
                  worker_accept_cb=worker_accept_cb)
        self._handler.job_available_cb(self, job)

    def generate_token(self, *, identity: str, room: str):
        identity = [uuid.uuid4(
        ) if self._handler.agent_identity_generator is None else self._handler.agent_identity_generator(room)]

        grants = api.VideoGrant(
            room=room,
            can_publish=True,
            can_subscribe=True,
            room_join=True,
            can_publish_data=True)
        t = api.AccessToken(api_key=self._api_key, api_secret=self._api_secret).with_identity(
            identity).with_grants(grants=grants)
        return t.to_jwt(ttl=3600)

    @abstractmethod
    async def _accept_job(self, job: Job):
        raise NotImplementedError()


class ManualWorker(Worker):
    async def _accept_job(self, job: Job):
        pass


class SFUTriggeredWorker(Worker):
    async def connect(self, *, ws_url: str, api_key: str, api_secret: str):
        pass

    async def disconnect(self):
        pass

    async def _accept_job(self, job: Job):
        # TODO: Tell the SFU we've accepted this job
        pass
