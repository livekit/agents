from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..llm import FunctionTool, RawFunctionTool


@runtime_checkable
class ToolSet(Protocol):
    def __livekit_tools__(self) -> Sequence[FunctionTool | RawFunctionTool]: ...
