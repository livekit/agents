import asyncio
import logging
import uuid
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
import livekit.api as api
import livekit.rtc as rtc
import livekit.agents._proto.livekit_agent_pb2 as proto


@dataclass
class JobContext:
    job: "Job"
    room: rtc.Room
    participant: Optional[rtc.RemoteParticipant]


@dataclass
class AgentParticipantInfo:
    identity: str
    metadata: str


class Job:

    def __init__(self,
                 *,
                 job_proto: proto.Job,
                 ws_url: str,
                 api_key: str,
                 api_secret: str,
                 participant_sid: Optional[str],
                 worker_accept_cb: Callable[["Job"], None]):
        self._worker_accept_cb = worker_accept_cb
        self._processors = []
        self._job_proto = job_proto
        self._room = rtc.Room()
        self._participant_sid = participant_sid
        self._api_key = api_key
        self._api_secret = api_secret
        self._ws_url = ws_url

    def link_processor(self, processor):
        self._processors.append(processor)

    async def accept(self, *, agent: Callable[[JobContext], Awaitable[None]], agent_participant_info: Optional[AgentParticipantInfo] = None):
        identity = agent_participant_info.identity if agent_participant_info is not None else f"agent-{self._job_proto.id}"
        token = self._generate_token(
            room=self._job_proto.room.name, identity=identity)
        await self._worker_accept_cb(self)

        try:
            await self._room.connect(url=self._ws_url, token=token, options=rtc.RoomOptions(auto_subscribe=True))
        except Exception as e:
            logging.error(
                "Error connecting to room, cancelling job.accept(): %s", e)
            raise e

        participant = None
        if self._participant_sid is not None:
            try:
                participant = await self._room.participants[self._participant_sid]
            except Exception as e:
                logging.error(
                    "Error getting participant '%s' - did they leave the room?, cancelling job.accept(): %s", self._participant_sid, e)
                try:
                    await self._room.disconnect()
                except Exception as room_disconnect_error:
                    logging.error(
                        "Error disconnecting from room after participant error: %s", room_disconnect_error)
                    raise room_disconnect_error
                raise e

        asyncio.create_task(
            agent(JobContext(job=self, room=self._room, participant=participant)))

    def _generate_token(self, room: str, identity: str):
        grants = api.VideoGrants(
            room=room,
            can_publish=True,
            can_subscribe=True,
            room_join=True,
            room_create=True,
            can_publish_data=True)
        t = api.AccessToken(api_key=self._api_key, api_secret=self._api_secret).with_identity(
            identity).with_grants(grants=grants)
        return t.to_jwt()
