import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import aiofiles
from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    utils,
)
from livekit.agents.voice import AgentTask, VoiceAgent
from livekit.agents.voice.events import AgentEvent, EventTypes
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
        agent: VoiceAgent | None,
        log: str | None,
        transcriptions_only: bool = False,
    ):
        """
        Initializes a ConversationPersistor instance which records the events and transcriptions of a VoiceAgent.

        Args:
            model (VoiceAgent): an instance of a VoiceAgent
            log (str): name of the external file to record events in
            transcriptions_only (bool): a boolean variable to determine if only transcriptions will be recorded, False by default
            user_transcriptions (arr): list of user transcriptions
            agent_transcriptions (arr): list of agent transcriptions
            events (arr): list of all events
            log_q (asyncio.Queue): a queue of EventLog and TranscriptionLog

        """
        super().__init__()

        self._agent = agent
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
    def model(self) -> VoiceAgent | None:
        return self._model

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

                    await file.write(
                        "\n" + log.time + " " + log.role + " " + log.transcription
                    )

    async def aclose(self) -> None:
        # Exits
        self._log_q.put_nowait(None)
        await self._main_task

    def start(self) -> None:
        # Listens for emitted VoiceAgent events
        self._main_task = asyncio.create_task(self._main_atask())

        @self._agent.on("user_started_speaking")
        def _user_started_speaking(ev: AgentEvent):
            event = EventLog(eventname=ev)
            self._log_q.put_nowait(event)

        @self._agent.on("user_stopped_speaking")
        def _user_stopped_speaking(ev: AgentEvent):
            event = EventLog(eventname=ev)
            self._log_q.put_nowait(event)

        @self._agent.on("user_input_transcribed")
        def _user_input_transcribed(ev: AgentEvent):
            if ev.is_final:
                event = EventLog(eventname="user_input_transcribed")
                self._log_q.put_nowait(event)
                transcription = TranscriptionLog(
                    role="user", transcription=ev.transcript
                )
                self._log_q.put_nowait(transcription)

        # @self._agent.on("agent_state_changed")
        # def _agent_state_changed(ev: AgentStateChangedEvent):
        #     """ The state is either "initializing", "listening", or "speaking" """
        #     name = "agent_state_changed: " + ev.state.value
        #     event = EventLog(name)
        #     self._log_q.put_nowait(event)

        # @self._agent.on("agent_started_speaking")
        # def _agent_started_speaking(ev: AgentEvent):
        #     event = EventLog("agent_started_speaking")
        #     self._log_q.put_nowait(event)

        # @self._agent.on("agent_stopped_speaking")
        # def _agent_stopped_speaking():
        #     event = EventLog("agent_stopped_speaking")
        #     self._log_q.put_nowait(event)


load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    agent = VoiceAgent(task=AgentTask(), llm=openai.realtime.RealtimeModel())

    cp = ConversationPersistor(model=agent, log="log.txt")
    cp.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    agent.start(ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
