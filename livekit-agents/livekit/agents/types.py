from dataclasses import dataclass
from typing import Literal, TypeVar, Union

from typing_extensions import TypeAlias

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

TOPIC_CHAT = "lk.chat"
TOPIC_TRANSCRIPTION = "lk.transcription"


_T = TypeVar("_T")


class NotGiven:
    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NotGivenOr: TypeAlias = Union[_T, NotGiven]
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
