from ... import llm
from ...llm.tool_context import ToolError, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask
from .task import Task


class TaskOrchestrator(AgentTask):
    def __init__(
        self,
        *,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        task_stack: list[Task] | None = None,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
    ):
        """Creates a TaskOrchestrator instance."""
        super().__init__(
            instructions="You are a task orchestrator managing a defined workflow. There is a stack of tasks, and if the user wants to regress to a previous question, call out_of_scope().",
            chat_ctx=chat_ctx,
            llm=llm,
        )
        self._task_stack = task_stack
        self._visited_tasks = {}
        self._task_results = {}
        self._main_atask = None
        self._current_task = None

    @property
    def task_stack(self) -> list:
        return self._task_stack

    @property
    def visited_tasks(self) -> dict:
        return self._visited_tasks

    @task_stack.setter
    def task_stack(self, task_stack: list) -> None:
        self._task_stack = task_stack

    def drain(self): ...  # or rename to aclose

    async def _main_task(self):
        while len(self._task_stack) > 0:
            task = self._task_stack.pop()

            current_agent_task = task.task
            current_tools = current_agent_task.tools
            current_tools.append(self._out_of_scope)
            await current_agent_task.update_tools(current_tools)

            result = await current_agent_task
            self._task_results[task] = result

            if task not in self._visited_tasks:
                self._visited_tasks[id(task)] = task

        self.complete(self._task_results)

    @function_tool
    async def out_of_scope(self, task_id: int):
        """Call to regress to a another question/task

        Args:
            task_id (int): The id of the task to regress to

        """
        if task_id in self._visited_tasks.keys():
            self._task_stack.append(self._visited_tasks[task_id])

        else:
            raise ToolError(
                "Unable to regress, requested task not found in previously visited tasks"
            )


# task results should be returned as:
# {Task1: Result1, Task2: Result2, ...}
