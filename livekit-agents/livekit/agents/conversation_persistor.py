from datetime import datetime
from dataclasses import dataclass

@dataclass
class EventLog:
    eventname: str | None 
    """name of recorded event"""
    time: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    """time the event is recorded"""

@dataclass
class TranscriptionLog:
    role: str | None    
    """role of the speaker"""
    transcription: str | None
    """transcription of speech"""
    time: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    """time the event is recorded"""

from livekit.agents import multimodal, utils
from livekit.agents.multimodal.multimodal_agent import EventTypes

class ConversationPersistor(utils.EventEmitter[EventTypes]):
    def __init__(
        self,
        *,
        model: multimodal.MultimodalAgent | None,
        log: str | None
    ):
        """
        Initializes a ConversationPersistor instance which records the events and transcriptions of a MultimodalAgent.

        Args:
            model (multimodal.MultimodalAgent): an instance of a MultiModalAgent
            log (str): name of the external file to record events in
        
        """
        super().__init__()

        self._model = model
        self._log = log

        self._user_transcriptions = []
        self._agent_transcriptions = []
        self._events = []

        self.start()

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
    
    def writelog(
            self,
            *, 
            event: EventLog | None = None, 
            transcription: TranscriptionLog | None = None
    ) -> None:
        """
        Writes events and transcriptions to an external file.

        Args:
            event (EventLog, optional): an instance of an EventLog
            transcription (TranscriptionLog, optional): an instance of a TranscriptionLog

        """
        with open(self._log, "a+") as file:
            if event:
                file.write("\n" + event.time + " " + event.eventname)
            if transcription:
                file.write("\n" + transcription.time + " " + transcription.role + " " + transcription.transcription)
            
    
    def start(self) -> None:
        @self._model.on("user_started_speaking")
        def _user_started_speaking():
            event = EventLog(eventname="user_started_speaking")
            self.writelog(event=event)

        @self._model.on("user_stopped_speaking")
        def _user_stopped_speaking():
            event = EventLog(eventname="user_stopped_speaking")
            self.writelog(event=event)
        
        @self._model.on("agent_started_speaking")
        def _agent_started_speaking():
            event = EventLog(eventname="agent_started_speaking")
            self.writelog(event=event)
        
        @self._model.on("agent_stopped_speaking")
        def _agent_stopped_speaking():
            transcription = TranscriptionLog(role="agent", transcription=(self._model._playing_handle._tr_fwd.played_text)[1:])
            self.writelog(transcription=transcription)

            event = EventLog(eventname="agent_stopped_speaking")
            self.writelog(event=event)

        from livekit.agents.llm import ChatMessage

        @self._model.on("user_speech_committed")
        def _user_speech_committed(user_msg: ChatMessage):
            transcription = TranscriptionLog(role="user", transcription=user_msg.content)
            self.writelog(transcription=transcription)

            event = EventLog(eventname="user_speech_committed")
            self.writelog(event=event)

        @self._model.on("agent_speech_committed")
        def _agent_speech_committed():
            event = EventLog(eventname="agent_speech_committed")
            self.writelog(event=event)

        @self._model.on("agent_speech_interrupted")
        def _agent_speech_interrupted():
            event = EventLog(eventname="agent_speech_interrupted")
            self.writelog(event=event)

        @self._model.on("function_calls_collected")
        def _function_calls_collected():
            event = EventLog(eventname="function_calls_collected")
            self.writelog(event=event)

        @self._model.on("function_calls_finished")
        def _function_calls_collected():
            event = EventLog(eventname="function_calls_finished")
            self.writelog(event=event)
