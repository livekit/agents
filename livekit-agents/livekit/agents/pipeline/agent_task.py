from ..llm import LLM, ChatContext, FunctionContext
from ..llm.function_context import (
    METADATA_ATTR,
    USE_DOCSTRING,
    FunctionInfo,
    ai_callable,
)
from ..stt import STT

# class TaskContext:
#     def __init__(self, assistant: "VoicePipelineAgent"):
#         self._assistant = assistant

#     @property
#     def agent(self) -> "VoicePipelineAgent":
#         return self._assistant

#     @property
#     def user_data(self) -> dict[str, Any]:
#         return self._assistant.user_data

#     @property
#     def current_task(self) -> "AgentTask" | None:
#         return self._assistant._current_task

#     @property
#     def room(self) -> rtc.Room:
#         if not hasattr(self._assistant, "_room"):
#             raise ValueError("VoicePipelineAgent is not started")
#         return self._assistant._room


class AgentTask:
    def __init__(
        self,
        instructions: str | None = None,
        fnc_ctx: FunctionContext | None = None,
        llm: LLM | None = None,
        stt: STT | None = None,
        name: str | None = None,
    ) -> None:
        self._chat_ctx = ChatContext()
        if instructions:
            self._chat_ctx.append(text=instructions, role="system")
        self._fnc_ctx = fnc_ctx
        self._llm = llm
        # TODO: support customized llm and stt
        self._stt = stt

        self._task_name = name or self.__class__.__name__

        # enter method for transition
        enter_fnc = self.enter
        if not hasattr(enter_fnc, METADATA_ATTR):
            enter_fnc = ai_callable(
                name=f"enter_{self._task_name}", description=USE_DOCSTRING
            )(self.enter)

        self._enter_fnc_info = FunctionContext._callable_to_fnc_info(enter_fnc)

    def can_enter(self) -> bool:
        return True

    def enter(self) -> "AgentTask" | tuple["AgentTask", str]:
        return self

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def enter_fnc_info(self) -> FunctionInfo:
        return self._enter_fnc_info

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> FunctionContext | None:
        return self._fnc_ctx

    @property
    def llm(self) -> LLM | None:
        return self._llm

    @property
    def stt(self) -> STT | None:
        return self._stt
