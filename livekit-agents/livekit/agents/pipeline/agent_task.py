import inspect
from typing import Callable, Dict, Optional, Type, Union

from ..llm import LLM, ChatContext, FunctionContext
from ..llm.function_context import METADATA_ATTR, ai_callable
from ..stt import STT


class AgentTask:
    # Single class-level storage for all tasks
    _registered_tasks: Dict[Union[str, Type["AgentTask"]], "AgentTask"] = {}

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

        self._fnc_ctx = FunctionContext()
        if functions:
            # register ai functions from the list
            for fnc in functions:
                if not hasattr(fnc, METADATA_ATTR):
                    fnc = ai_callable()(fnc)
                self._fnc_ctx._register_ai_function(fnc)

        # register ai functions from the class
        for _, member in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(member, METADATA_ATTR):
                self._fnc_ctx._register_ai_function(member)

        self._llm = llm
        self._stt = None

        # Auto-register if name is provided
        if name is not None:
            self.register_task(self, name)

    @classmethod
    def register_task(
        cls, task: "AgentTask", name: Optional[str] = None
    ) -> "AgentTask":
        """Register a task instance globally"""
        # Register by name if provided
        if name is not None:
            if name in cls._registered_tasks:
                raise ValueError(f"Task with name '{name}' already registered")
            cls._registered_tasks[name] = task

        # Always register by type
        task_type = type(task)
        if task_type in cls._registered_tasks:
            raise ValueError(f"Task of type {task_type.__name__} already registered")
        cls._registered_tasks[task_type] = task

        return task

    def inject_chat_ctx(self, chat_ctx: ChatContext) -> None:
        self._chat_ctx.messages.extend(chat_ctx.messages)

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> FunctionContext:
        return self._fnc_ctx

    @property
    def llm(self) -> Optional[LLM]:
        return self._llm

    @property
    def stt(self) -> Optional[STT]:
        return self._stt

    @classmethod
    def get_task(cls, key: Union[str, Type["AgentTask"]]) -> "AgentTask":
        """Get task instance by name or class"""
        if key not in cls._registered_tasks:
            raise ValueError(f"Task with name or class {key} not found")
        return cls._registered_tasks[key]

    @classmethod
    def all_registered_tasks(cls) -> list["AgentTask"]:
        """Get all registered tasks"""
        return list(set(cls._registered_tasks.values()))
