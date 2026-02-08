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
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    EventTypes,
    FunctionToolsExecutedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.plugins import cartesia, deepgram, openai, silero


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
        transcriptions of an AgentSession.

        Args:
            session (AgentSession): an instance of an AgentSession
            log (str): name of the external file to record events in
            transcriptions_only (bool): a boolean variable to determine
                if only transcriptions will be recorded, False by default
            user_transcriptions (arr): list of user transcriptions
            agent_transcriptions (arr): list of agent transcriptions
            events (arr): list of all events
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

        @self.session.on("user_state_changed")
        def _user_state_changed(ev: UserStateChangedEvent):
            """
            The state is either: "speaking", "listening", or "away"
            """
            name = ev.type + " to " + ev.new_state + " from " + ev.old_state
            event = EventLog(eventname=name)
            self._log_q.put_nowait(event)

        @self.session.on("user_input_transcribed")
        def _user_input_transcribed(ev: UserInputTranscribedEvent):
            if ev.is_final:
                event = EventLog(eventname=ev.type)
                self._log_q.put_nowait(event)
                transcription = TranscriptionLog(role="user", transcription=ev.transcript)
                self._log_q.put_nowait(transcription)

        @self.session.on("conversation_item_added")
        def _conversation_item_added(ev: ConversationItemAddedEvent):
            if ev.item.role == "assistant":
                transcription = TranscriptionLog(
                    role="assistant", transcription=ev.item.text_content
                )
                self._log_q.put_nowait(transcription)
            event = EventLog(eventname=ev.type)
            self._log_q.put_nowait(event)

        @self.session.on("agent_state_changed")
        def _agent_state_changed(ev: AgentStateChangedEvent):
            """
            The state is either: "initializing", "idle", "listening", "thinking", or "speaking"
            """
            if (
                self.session.current_speech is not None
                and self.session.current_speech.interrupted
                and ev.old_state == "speaking"
            ):
                event = EventLog(eventname="speech_interrupted")
                self._log_q.put_nowait(event)
            name = ev.type + " to " + ev.new_state + " from " + ev.old_state
            event = EventLog(eventname=name)
            self._log_q.put_nowait(event)

        @self.session.on("speech_created")
        def _speech_created(ev: SpeechCreatedEvent):
            name = ev.type + " " + ev.source
            event = EventLog(eventname=name)
            self._log_q.put_nowait(event)

        @self.session.on("function_tools_executed")
        def _function_tools_executed(ev: FunctionToolsExecutedEvent):
            function_calls = ev.function_calls
            for function in function_calls:
                name = function.call_id + " " + function.name + " args: " + function.arguments
                event = EventLog(eventname=name)
                self._log_q.put_nowait(event)
            function_outputs = ev.function_call_outputs
            for output in function_outputs:
                name = function.call_id + " " + "function_output " + output.output
                event = EventLog(eventname=name)
                self._log_q.put_nowait(event)

        @self.session.on("error")
        def _error_(ev: ErrorEvent):
            event = EventLog(eventname=ev.error.type)
            self._log_q.put_nowait(event)


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a helpful assistant that can answer questions and help with
                            tasks. The conversation will be recorded.""",
        )

    @function_tool()
    async def open_door(self) -> str:
        if self.session.current_speech is not None:
            await self.session.current_speech
        await self.session.generate_reply(instructions="Opening the door..", tool_choice="none")

        return "The door is open!"


load_dotenv()

logger = logging.getLogger("my-persistor")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        stt=deepgram.STT(), llm=openai.LLM(), tts=cartesia.TTS(), vad=silero.VAD.load()
    )

    cp = ConversationPersistor(session=session, log="log.txt")
    cp.start()

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
