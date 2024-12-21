from typing import TYPE_CHECKING, Callable, Optional, Union

from ..llm import LLM, ChatContext, FunctionContext
from ..llm.function_context import USE_DOCSTRING, FunctionInfo, ai_callable
from ..stt import STT

if TYPE_CHECKING:
    from ..pipeline import VoicePipelineAgent


class AgentTask:
    def __init__(
        self,
        instructions: Optional[str] = None,
        functions: Optional[list[Callable]] = None,
        llm: Optional[LLM] = None,
        name: Optional[str] = None,
    ) -> None:
        self._chat_ctx = ChatContext()
        if instructions:
            self._chat_ctx.append(text=instructions, role="system")
        self._fnc_ctx: Optional[FunctionContext] = None
        if functions:
            self._fnc_ctx = FunctionContext()
            for fnc in functions:
                self._fnc_ctx._register_ai_function(fnc)

        self._llm = llm
        self._stt = None
        self._name = name or self.__class__.__name__
        self._enter_fnc_info = FunctionContext._callable_to_fnc_info(self.enter)
        if not self._enter_fnc_info:
            raise ValueError("enter function must be decorated with ai_callable")

    def can_enter(self, agent: "VoicePipelineAgent") -> bool:
        return True

    @ai_callable(name="transfer_to_task", description=USE_DOCSTRING)
    async def enter(self) -> Union["AgentTask", tuple["AgentTask", str]]:
        """Called to enter the task."""
        return self

    @property
    def name(self) -> str:
        return self._name

    @property
    def enter_fnc_info(self) -> FunctionInfo:
        return self._enter_fnc_info  # type: ignore

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> Optional[FunctionContext]:
        return self._fnc_ctx

    @property
    def llm(self) -> Optional[LLM]:
        return self._llm

    @property
    def stt(self) -> Optional[STT]:
        return self._stt
