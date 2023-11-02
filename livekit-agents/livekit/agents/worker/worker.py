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

    def __init__(self, *,
                 ws_url: str,
                 api_key: str,
                 api_secret: str,
                 job_available_cb: Callable[["Worker", Job], Awaitable[None]]):
        self._active_jobs: [Job] = []
        self._ws_url = ws_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._job_available_cb = job_available_cb

    async def simulate_job(self, *, job_type: proto.JobType, room: str, participant_sid: Optional[str] = None):
        job_id = f"simulated-{uuid.uuid4()}"
        proto_room = model_proto.Room(sid="", name=room, empty_timeout=0, max_participants=0, creation_time=0, turn_password="",
                                      enabled_codecs=None, metadata="", num_participants=0, num_publishers=0, active_recording=False)
        job = proto.Job(id=job_id, type=job_type,
                        room=proto_room, participant=None)

        if job_type == proto.JobType.JT_ROOM:
            if participant_sid is not None:
                raise ValueError("participant_sid must be None for JT_ROOM")
        elif job_type == proto.JobType.JT_PARTICIPANT:
            if participant_sid is None:
                raise ValueError(
                    "participant_sid must be provided for JT_PARTICIPANT")
            job.participant = model_proto.ParticipantInfo()

        await self._on_job_available(job)

    async def _on_job_available(self, job_proto: proto.Job):
        async def worker_accept_cb(job: Job):
            await self._accept_job(job)
            self._active_jobs.append(job)

        job = Job(job_proto=job_proto,
                  api_key=self._api_key,
                  api_secret=self._api_secret,
                  ws_url=self._ws_url,
                  participant_sid=job_proto.participant.sid if job_proto.type == proto.JobType.JT_PARTICIPANT else None,
                  worker_accept_cb=worker_accept_cb)
        await self._job_available_cb(self, job)

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
