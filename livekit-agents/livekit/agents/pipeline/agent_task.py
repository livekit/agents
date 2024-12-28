from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Union

from ..llm import LLM, ChatContext, ChatMessage, FunctionContext
from ..llm.function_context import FunctionInfo, _UseDocMarker, ai_callable
from ..stt import STT

if TYPE_CHECKING:
    from ..pipeline import VoicePipelineAgent


BeforeEnterCallback = Callable[
    ["VoicePipelineAgent", "AgentTask"],
    Awaitable[Union["AgentTask", tuple["AgentTask", str]]],
]


def _get_last_n_messages(messages: list[ChatMessage], n: int) -> list[ChatMessage]:
    collected_messages = messages.copy()[-n:]
    while collected_messages and collected_messages[0].role in ["system", "tool"]:
        collected_messages.pop(0)
    return collected_messages


async def _default_before_enter_cb(
    agent: "VoicePipelineAgent", task: "AgentTask"
) -> tuple["AgentTask", str]:
    # keep the last n messages for the next stage
    keep_last_n = 6
    previous_messages = _get_last_n_messages(agent.chat_ctx.messages, keep_last_n)
    task.chat_ctx.messages.extend(previous_messages)

    message = f"Transferred from {agent.current_agent_task.name} to {task.name}."
    return task, message


def _default_can_enter_cb(agent: "VoicePipelineAgent") -> bool:
    return True


@dataclass(frozen=True)
class AgentTaskOptions:
    can_enter_cb: Callable[["VoicePipelineAgent"], bool] = _default_can_enter_cb
    """callback to check if the task can be entered"""
    transfer_function_description: Optional[Union[str, _UseDocMarker]] = None
    """description of the transfer function, use `Called to transfer to {task_name}` if not provided"""
    before_enter_cb: BeforeEnterCallback = _default_before_enter_cb
    """callback to call before entering the task"""


class AgentTask:
    def __init__(
        self,
        name: Optional[str] = None,
        instructions: Optional[str] = None,
        functions: Optional[list[Callable]] = None,
        llm: Optional[LLM] = None,
        options: AgentTaskOptions = AgentTaskOptions(),
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

        self._task_name = name or self.__class__.__name__
        self._opts = options

        # transfer function
        from ..pipeline import AgentCallContext

        transfer_fnc_desc = (
            options.transfer_function_description
            if options.transfer_function_description is not None
            else f"Called to transfer to {self._task_name}"
        )

        @ai_callable(
            name=f"transfer_to_{self._task_name}", description=transfer_fnc_desc
        )
        async def transfer_fnc() -> Union["AgentTask", tuple["AgentTask", str]]:
            agent = AgentCallContext.get_current().agent
            return await self._opts.before_enter_cb(agent, self)

        self._transfer_fnc_info = FunctionContext._callable_to_fnc_info(transfer_fnc)

    def _can_enter(self, agent: "VoicePipelineAgent") -> bool:
        return self._opts.can_enter_cb(agent)

    @property
    def name(self) -> str:
        return self._task_name

    @property
    def transfer_fnc_info(self) -> FunctionInfo:
        return self._transfer_fnc_info  # type: ignore

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
