import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import aiofiles
from dotenv import load_dotenv

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    utils,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.events import EventTypes
from livekit.plugins import openai


@dataclass
class EventLog:
    eventname: str | None
    """name of recorded event"""
    time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    """time the event is recorded"""


@dataclass
class TranscriptionLog:
    role: str | None
    """role of the speaker"""
    transcription: str | None
    """transcription of speech"""
    time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    """time the event is recorded"""


class ConversationPersistor(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        session: AgentSession | None,
        log: str | None,
        transcriptions_only: bool = False,
    ):
        """
        Initializes a ConversationPersistor instance which records the events and
        transcriptions of a Realtime Agent Session.

        Args:
            session (AgentSession): an instance of an AgentSession
            log (str): name of the external file to record events in
            transcriptions_only (bool): a boolean variable to determine
                if only transcriptions will be recorded, False by default
            user_transcriptions (arr): list of user transcriptions
            agent_transcriptions (arr): list of agent transcriptions
            events (arr): list of all events
            log_q (asyncio.Queue): a queue of EventLog and TranscriptionLog

        """
        super().__init__()

        self._session = session
        self._log = log
        self._transcriptions_only = transcriptions_only

        self._user_transcriptions = []
        self._agent_transcriptions = []
        self._events = []

        self._log_q = asyncio.Queue[Union[EventLog, TranscriptionLog, None]]()

    @property
    def log(self) -> str | None:
        return self._log

    @property
    def session(self) -> AgentSession | None:
        return self._session

    @property
    def user_transcriptions(self) -> dict:
        return self._user_transcriptions

    @property
    def agent_transcriptions(self) -> dict:
        return self._agent_transcriptions

    @property
    def events(self) -> dict:
        return self._events

    @log.setter
    def log(self, newlog: str | None) -> None:
        self._log = newlog

    async def _main_atask(self) -> None:
        # Writes to file asynchronously
        while True:
            log = await self._log_q.get()

            if log is None:
                break

            async with aiofiles.open(self._log, "a") as file:
                if type(log) is EventLog and not self._transcriptions_only:
                    self._events.append(log)
                    await file.write("\n" + log.time + " " + log.eventname)

                if type(log) is TranscriptionLog:
                    if log.role == "user":
                        self._user_transcriptions.append(log)
                    else:
                        self._agent_transcriptions.append(log)

                    await file.write("\n" + log.time + " " + log.role + " " + log.transcription)

    async def aclose(self) -> None:
        # Exits
        self._log_q.put_nowait(None)
        await self._main_task

    def start(self) -> None:
        # Listens for emitted events
        self._main_task = asyncio.create_task(self._main_atask())

        @self.session.current_agent.realtime_llm_session.on("openai_server_event_received")
        def _server_event_received(event: dict):
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                transcript = TranscriptionLog(role="user", transcription=event["transcript"])
                self._log_q.put_nowait(transcript)

            elif event["type"] == "response.audio_transcript.done":
                transcript = TranscriptionLog(role="assistant", transcription=event["transcript"])
                self._log_q.put_nowait(transcript)

            elif event["type"] == "response.function_call_arguments.done":
                eventname = event["type"] + " " + event["arguments"]
                ev = EventLog(eventname=eventname)
                self._log_q.put_nowait(ev)

            elif event["type"] == "rate_limits.updated":
                eventname = (
                    event["type"]
                    + " "
                    + "remaining_tokens: "
                    + str(event["rate_limits"][1]["remaining"])
                )
                ev = EventLog(eventname=eventname)
                self._log_q.put_nowait(ev)

            elif event["type"] == "error":
                eventname = (
                    event["type"] + " " + event["error"]["type"] + " " + event["error"]["message"]
                )
                ev = EventLog(eventname=eventname)
                self._log_q.put_nowait(ev)

            elif ".delta" not in event["type"]:  # reduces spamming of audio bytes
                ev = EventLog(eventname=event["type"])
                self._log_q.put_nowait(ev)

        @self.session.current_agent.realtime_llm_session.on("openai_client_event_queued")
        def _client_event_queued(event: dict):
            if "audio" not in event["type"]:
                eventname = event["type"]
                ev = EventLog(eventname=eventname)
                self._log_q.put_nowait(ev)


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a helpful assistant that can answer questions and help with
                            tasks. The conversation will be recorded.""",
        )

    @function_tool()
    async def open_door(self) -> str:
        await self.session.generate_reply(instructions="Opening the door..")

        return "The door is open!"


load_dotenv()

logger = logging.getLogger("realtime-persistor")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(llm=openai.realtime.RealtimeModel())
    cp = ConversationPersistor(session=session, log="log.txt")

    await session.start(agent=MyAgent(), room=ctx.room)
    cp.start()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
