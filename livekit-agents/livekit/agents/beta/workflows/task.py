from collections.abc import Callable

from ...llm import ChatContext
from ...voice.agent import AgentTask


class Task:
    def __init__(self, task_factory: Callable[[], AgentTask], *, description: str) -> None:
        """Creates a Task instance which holds an AgentTask and its description.

        Args:
            task_factory (Callable[[],AgentTask]): Generator of AgentTask (ex. lambda: GetEmailTask())
            description (str): A description of the AgentTask
        """
        self._task_factory = task_factory
        self._description = description
        self._task = None
        self._saved_chat_ctx = ChatContext()

    @property
    def description(self) -> str:
        return self._description

    async def create_new_task(self) -> AgentTask:
        if self._task:
            self._saved_chat_ctx = self._task.chat_ctx
        self._task = self._task_factory()
        await self._task.update_chat_ctx(self._saved_chat_ctx)
        return self._task
