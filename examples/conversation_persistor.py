import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import aiofiles
from dotenv import load_dotenv

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, multimodal, utils
from livekit.agents.llm import ChatMessage
from livekit.agents.multimodal.multimodal_agent import EventTypes
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
        model: multimodal.MultimodalAgent | None,
        log: str | None,
        transcriptions_only: bool = False,
    ):
        """
        Initializes a ConversationPersistor instance which records the events and transcriptions of a MultimodalAgent.

        Args:
            model (multimodal.MultimodalAgent): an instance of a MultiModalAgent
            log (str): name of the external file to record events in
            transcriptions_only (bool): a boolean variable to determine if only transcriptions will be recorded, False by default
            user_transcriptions (arr): list of user transcriptions
            agent_transcriptions (arr): list of agent transcriptions
            events (arr): list of all events
            log_q (asyncio.Queue): a queue of EventLog and TranscriptionLog

        """
        super().__init__()

        self._model = model
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
    def model(self) -> multimodal.MultimodalAgent | None:
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

                    await file.write("\n" + log.time + " " + log.role + " " + log.transcription)

    async def aclose(self) -> None:
        # Exits
        self._log_q.put_nowait(None)
        await self._main_task

    def start(self) -> None:
        # Listens for emitted MultimodalAgent events
        self._main_task = asyncio.create_task(self._main_atask())

        @self._model.on("user_started_speaking")
        def _user_started_speaking():
            event = EventLog(eventname="user_started_speaking")
            self._log_q.put_nowait(event)

        @self._model.on("user_stopped_speaking")
        def _user_stopped_speaking():
            event = EventLog(eventname="user_stopped_speaking")
            self._log_q.put_nowait(event)

        @self._model.on("agent_started_speaking")
        def _agent_started_speaking():
            event = EventLog(eventname="agent_started_speaking")
            self._log_q.put_nowait(event)

        @self._model.on("agent_stopped_speaking")
        def _agent_stopped_speaking():
            transcription = TranscriptionLog(
                role="agent",
                transcription=(self._model._playing_handle._tr_fwd.played_text)[1:],
            )
            self._log_q.put_nowait(transcription)

            event = EventLog(eventname="agent_stopped_speaking")
            self._log_q.put_nowait(event)

        @self._model.on("user_speech_committed")
        def _user_speech_committed(user_msg: ChatMessage):
            transcription = TranscriptionLog(role="user", transcription=user_msg.content)
            self._log_q.put_nowait(transcription)

            event = EventLog(eventname="user_speech_committed")
            self._log_q.put_nowait(event)

        @self._model.on("agent_speech_committed")
        def _agent_speech_committed():
            event = EventLog(eventname="agent_speech_committed")
            self._log_q.put_nowait(event)

        @self._model.on("agent_speech_interrupted")
        def _agent_speech_interrupted():
            event = EventLog(eventname="agent_speech_interrupted")
            self._log_q.put_nowait(event)

        @self._model.on("function_calls_collected")
        def _function_calls_collected():
            event = EventLog(eventname="function_calls_collected")
            self._log_q.put_nowait(event)

        @self._model.on("function_calls_finished")
        def _function_calls_finished():
            event = EventLog(eventname="function_calls_finished")
            self._log_q.put_nowait(event)


load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            voice="alloy",
            temperature=0.8,
            instructions="You are a helpful assistant.",
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
            ),
        ),
    )

    cp = ConversationPersistor(model=agent, log="log.txt")
    cp.start()

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    agent.start(ctx.room, participant)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
