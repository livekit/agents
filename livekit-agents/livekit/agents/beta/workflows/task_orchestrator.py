from ... import llm
from ...llm.tool_context import ToolError, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask


class TaskOrchestrator(AgentTask):
    def __init__(
        self,
        *,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        task_stack: list | None = None,
        llm,
    ):
        """Creates a TaskOrchestrator instance."""
        # pass task stack list in instructions for context to regress
        super().__init__(
            instructions="You are a task orchestrator managing a defined workflow. There is a stack of tasks, and if the user wants to regress to a previous question, call regress_to_task()."
        )
        self._task_stack = []
        self._visited_tasks = []
        self._task_results = []
        self._main_atask = None
        self._current_task = None

    @property
    def task_stack(self) -> list:
        return self._task_stack

    @property
    def visited_tasks(self) -> list:
        return self._visited_tasks

    @task_stack.setter
    def task_stack(self, task_stack: list) -> None:
        self._task_stack = task_stack

    def drain(self): ...

    async def _main_task(self):
        while len(self._task_stack) > 0:
            currtask = self._task_stack.pop()

            if currtask in self._visited_tasks:
                # TODO pass last user message as the trigger
                currtask = currtask.regress(trigger="")
            result = await currtask()
            self._task_results.append(result)

            if currtask not in self._visited_tasks:
                self._visited_tasks.append(currtask)
        self.complete(self._task_results)

    @function_tool
    async def regress_to_task(self, task_name: str):
        """Call to regress to a previous question/task

        Args:
            task_name (str): The name of the task to regress to

        """
        for task in self._visited_tasks:
            if task.___class__.__name__ == task_name:
                self._task_stack.append(task)
        else:
            raise ToolError(f"Unable to regress, {task_name} not found in previously visited tasks")
