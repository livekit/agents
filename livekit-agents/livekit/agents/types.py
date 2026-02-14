from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID = "lk.segment_id"
ATTRIBUTE_TRANSCRIPTION_TRACK_ID = "lk.transcribed_track_id"
ATTRIBUTE_TRANSCRIPTION_FINAL = "lk.transcription_final"
ATTRIBUTE_PUBLISH_ON_BEHALF = "lk.publish_on_behalf"
"""
The identity of the agent participant that an avatar worker is publishing on behalf of.
"""
ATTRIBUTE_AGENT_STATE = "lk.agent.state"
"""
The state of the agent, stored in the agent's attributes.
This can be retrieved on the client side by using `RemoteParticipant.attributes`.

With components-js, this can be easily retrieved using:

```js
const { state, ... } = useVoiceAssistant();
```
"""

ATTRIBUTE_AGENT_NAME = "lk.agent.name"
"""
The name of the agent, stored in the agent's attributes.
This is set when the agent joins a room and can be used to identify the agent type.
"""

ATTRIBUTE_SIMULATOR = "lk.simulator"
"""
Indicates that the participant is a simulator for testing purposes.
When set to "true", the agent will skip audio input/output processing.
"""

TOPIC_CHAT = "lk.chat"
TOPIC_TRANSCRIPTION = "lk.transcription"
TOPIC_CLIENT_EVENTS = "lk.agent.events"
"""
Topic for streaming agent events to room participants.
"""

RPC_GET_SESSION_STATE = "lk.agent.get_session_state"
"""
RPC method to get the current session state.
"""

RPC_GET_CHAT_HISTORY = "lk.agent.get_chat_history"
"""
RPC method to get the agent<>user conversation turns.
"""

RPC_GET_AGENT_INFO = "lk.agent.get_agent_info"
"""
RPC method to get information about the current agent.
"""

RPC_SEND_MESSAGE = "lk.agent.send_message"
"""
RPC method to send a message and get the agent's response.
"""

TOPIC_AGENT_REQUEST = "lk.agent.request"
"""
Topic for sending requests to the agent via text streams (no size limit).
"""

TOPIC_AGENT_RESPONSE = "lk.agent.response"
"""
Topic for receiving responses from the agent via text streams (no size limit).
"""

USERDATA_TIMED_TRANSCRIPT = "lk.timed_transcripts"
"""
The key for the timed transcripts in the audio frame userdata.
"""


_T = TypeVar("_T")


class FlushSentinel:
    pass


class NotGiven:
    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.is_instance_schema(cls)


NotGivenOr: TypeAlias = _T | NotGiven
NOT_GIVEN = NotGiven()


@dataclass(frozen=True)
class APIConnectOptions:
    max_retry: int = 3
    """
    Maximum number of retries to connect to the API.
    """

    retry_interval: float = 2.0
    """
    Interval between retries to connect to the API in seconds.
    """

    timeout: float = 10.0
    """
    Timeout for connecting to the API in seconds.
    """

    def __post_init__(self) -> None:
        if self.max_retry < 0:
            raise ValueError("max_retry must be greater than or equal to 0")

        if self.retry_interval < 0:
            raise ValueError("retry_interval must be greater than or equal to 0")

        if self.timeout < 0:
            raise ValueError("timeout must be greater than or equal to 0")

    def _interval_for_retry(self, num_retries: int) -> float:
        """
        Return the interval for the given number of retries.

        The first retry is immediate, and then uses specified retry_interval
        """
        if num_retries == 0:
            return 0.1
        return self.retry_interval


DEFAULT_API_CONNECT_OPTIONS = APIConnectOptions()


class TimedString(str):
    """A string with optional start and end timestamps for word-level alignment."""

    start_time: NotGivenOr[float]
    end_time: NotGivenOr[float]
    confidence: NotGivenOr[float]
    start_time_offset: NotGivenOr[float]
    # offset relative to the start of the audio input stream or session in seconds, used in STT plugins

    def __new__(
        cls,
        text: str,
        start_time: NotGivenOr[float] = NOT_GIVEN,
        end_time: NotGivenOr[float] = NOT_GIVEN,
        confidence: NotGivenOr[float] = NOT_GIVEN,
        start_time_offset: NotGivenOr[float] = NOT_GIVEN,
    ) -> "TimedString":
        obj = super().__new__(cls, text)
        obj.start_time = start_time
        obj.end_time = end_time
        obj.confidence = confidence
        obj.start_time_offset = start_time_offset
        return obj
