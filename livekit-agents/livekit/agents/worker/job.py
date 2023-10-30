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
            await self._room.connect(url=self._ws_url, token=self._token, options=rtc.RoomOptions(auto_subscribe=True))
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

        asyncio.create_task(agent(Job.AgentParams(
            room=self._room, participant=participant)))
