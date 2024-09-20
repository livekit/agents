from typing import Literal, Union

ATTR_AGENT_STATE = "lk.agent.state"

AgentState = Union[Literal["initializing", "listening", "thinking", "speaking"], str]
