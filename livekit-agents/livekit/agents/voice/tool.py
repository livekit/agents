from typing import Protocol

from ..llm import FunctionTool, RawFunctionTool


class Tool(Protocol):
    def __livekit_tools__(self) -> list[FunctionTool | RawFunctionTool]: ...
