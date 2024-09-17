from typing import Literal, Union

AgentState = Union[Literal["initializing", "listening", "thinking", "speaking"], str]
