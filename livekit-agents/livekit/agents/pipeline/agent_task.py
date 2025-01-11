import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type, Union

from ..llm import LLM, ChatContext, FunctionContext
from ..llm.function_context import METADATA_ATTR, ai_callable
from ..stt import STT
from .speech_handle import SpeechHandle

logger = logging.getLogger(__name__)


class ResultNotSetError(Exception):
    """Exception raised when the task result is not set."""


class SilentSentinel:
    """Sentinel value to indicate the function call shouldn't create a response."""

    def __init__(self, result: Any = None, error: Optional[Exception] = None):
        self._result = result
        self._error = error

    def __repr__(self) -> str:
        return f"SilentSentinel(result={self._result}, error={self._error})"


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
        self._name = name

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name})"


class AgentInlineTask(AgentTask):
    def __init__(
        self,
        instructions: Optional[str] = None,
        functions: Optional[list[Callable]] = None,
        llm: Optional[LLM] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(instructions, functions, llm, name)

        self._done_fut: asyncio.Future[None] = asyncio.Future()
        self._result: Optional[Any] = None
        self._error: Optional[Exception] = None

        self._parent_task: AgentTask | None = None
        self._parent_speech: SpeechHandle | None = None

    async def run(self, proactive_reply: bool = True) -> Any:
        from ..pipeline.pipeline_agent import AgentCallContext

        call_ctx = AgentCallContext.get_current()
        agent = call_ctx.agent

        self._parent_task = agent.current_agent_task
        self._parent_speech = call_ctx.speech_handle
        agent.update_task(self)
        logger.debug(
            "running inline task",
            extra={"task": str(self), "parent_task": str(self._parent_task)},
        )
        # generate reply to the user
        if proactive_reply:
            speech_handle = SpeechHandle.create_assistant_speech(
                allow_interruptions=agent._opts.allow_interruptions,
                add_to_chat_ctx=True,
            )
            self._proactive_reply_task = asyncio.create_task(
                agent._synthesize_answer_task(None, speech_handle)
            )
            if self._parent_speech is not None:
                self._parent_speech.add_nested_speech(speech_handle)
            else:
                agent._add_speech_for_playout(speech_handle)

        # wait for the task to complete
        try:
            await self._done_fut
            if self._error:
                raise self._error

            if self._result is None:
                raise ResultNotSetError()
            return self._result
        finally:
            # reset the parent task
            agent.update_task(self._parent_task)
            logger.debug(
                "inline task completed",
                extra={
                    "result": self._result,
                    "error": self._error,
                    "task": str(self),
                    "parent_task": str(self._parent_task),
                },
            )

    @ai_callable()
    def comfirm_result(self) -> SilentSentinel:
        """Called when user comfirms the information is correct or user wants to exit the task.
        Double check with the user before calling this function. This function should be called last in the task."""
        if not self._done_fut.done():
            self._done_fut.set_result(None)
        return SilentSentinel()

    @property
    def done(self) -> bool:
        return self._done_fut.done()

    @property
    def result(self) -> Any:
        return self._result

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    def __repr__(self) -> str:
        speech_id = self._parent_speech.id if self._parent_speech else None
        return (
            f"{self.__class__.__name__}(name={self._name}, parent_speech={speech_id})"
        )
