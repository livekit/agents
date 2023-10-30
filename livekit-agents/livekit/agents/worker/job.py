import asyncio
import logging
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
import livekit.api as api
import livekit.rtc as rtc
import livekit.agents._proto.livekit_agent_pb2 as proto


class Job:

    @dataclass
    class AgentParams:
        room: rtc.Room
        participant: Optional[rtc.RemoteParticipant]

    def __init__(self,
                 *,
                 job_id: str,
                 ws_url: str,
                 token: str,
                 participant_sid: Optional[str],
                 worker_accept_cb: Callable[["Job"], None]):
        self._worker_accept_cb = worker_accept_cb
        self._job_id = job_id
        self._room = rtc.Room()
        self._participant_sid = participant_sid
        self._ws_url = ws_url
        self._token = token

    async def accept(self, agent: Callable[[AgentParams], Awaitable[None]]):
        await self._worker_accept_cb(self)

        try:
            await self._room.connect(url=self._ws_url, token=self._token, options=rtc.RoomOptions(auto_subscribe=False))
        except Exception as e:
            logging.error(
                "Error connecting to room, cancelling job.accept(): %s", e)
            raise e

        asyncio.create_task(agent(Job.AgentParams(
            room=self._room, participant=self._participant_sid)))
