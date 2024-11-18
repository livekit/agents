from dataclasses import dataclass
from typing import Literal, TypeVar, Union

AgentState = Union[Literal["initializing", "listening", "thinking", "speaking"], str]
ATTRIBUTE_AGENT_STATE = "lk.agent.state"
"""
The state of the agent, stored in the agent's attributes.
This can be retrieved on the client side by using `RemoteParticipant.attributes`.

With components-js, this can be easily retrieved using:

```js
const { state, ... } = useVoiceAssistant();
```
"""


_T = TypeVar("_T")


class NotGiven:
    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NotGivenOr = Union[_T, NotGiven]
NOT_GIVEN = NotGiven()


@dataclass(frozen=True)
class APIConnectOptions:
    max_retry: int = 3
    """
    Maximum number of retries to connect to the API.
    """

    retry_interval: float = 5.0
    """
    Interval between retries to connect to the API in seconds.
    """

    timeout: float = 10.0
    """
    Timeout for connecting to the API in seconds.
    """

    def __post_init__(self):
        if self.max_retry < 0:
            raise ValueError("max_retry must be greater than or equal to 0")

        if self.retry_interval < 0:
            raise ValueError("retry_interval must be greater than or equal to 0")

        if self.timeout < 0:
            raise ValueError("timeout must be greater than or equal to 0")


DEFAULT_API_CONNECT_OPTIONS = APIConnectOptions()
