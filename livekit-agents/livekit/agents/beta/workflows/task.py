from collections.abc import Callable

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

    @property
    def task(self) -> AgentTask:
        return self._task_factory()

    @property
    def description(self) -> str:
        return self._description
