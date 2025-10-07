import json
from collections.abc import Callable
from typing import Annotated, Any, TypeAlias

from pydantic import Field

from ... import FunctionTool, llm
from ...llm import ChatContext
from ...llm.tool_context import ToolError, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask


class Task:
    def __init__(self, task_factory: Callable[[], AgentTask], *, id: str, description: str) -> None:
        """Creates a Task instance which holds an AgentTask and its description.

        Args:
            task_factory (Callable[[],AgentTask]): Generator of AgentTask (ex. lambda: GetEmailTask())
            id (str): The ID of the AgentTask, ex. "get_email_task"
            description (str): A description of the AgentTask
        """
        self._task_factory = task_factory
        self._id = id
        self._description = description
        self._task = None
        self._saved_chat_ctx = ChatContext()

    @property
    def id(self) -> str:
        return self._id

    @property
    def description(self) -> str:
        return self._description

    async def create_new_task(self) -> AgentTask:
        if self._task:
            self._saved_chat_ctx = self._task.chat_ctx
        self._task = self._task_factory()
        await self._task.update_chat_ctx(self._saved_chat_ctx)
        return self._task


ResultT: TypeAlias = dict[str, Any]


class TaskOrchestrator(AgentTask[ResultT]):
    def __init__(
        self,
        tasks: list[Task] | None = None,
        *,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
    ):
        """Creates a TaskOrchestrator instance."""
        super().__init__(
            instructions="*empty*",
            chat_ctx=chat_ctx,
            llm=None,
        )
        self._task_stack = tasks
        self._task_order = tasks[::-1]
        # TODO restructure task id lookup for out_of_scope function
        self._task_ids = [id(task) for task in self._task_order]
        self._task_descriptions = [task.description for task in self._task_order]
        self._task_id_lookup = dict(zip(self._task_descriptions, self._task_ids))
        self._out_of_scope_func = self.construct_out_of_scope_tool(
            task_id_lookup=self._task_id_lookup
        )
        self._out_of_scope_func.__doc__ = f"""Call to regress to another task according to what the user requested to modify, return the corresponding task id. The following are the IDs and their corresponding task description. {json.dumps(self._task_id_lookup)}"""
        self._visited_tasks = {}
        self._task_results = {}

        self._main_atask = None
        self._current_task = None

    @property
    def _task_stack(self) -> list:
        return self._task_stack

    @property
    def _visited_tasks(self) -> dict:
        return self._visited_tasks

    @_task_stack.setter
    def _task_stack(self, task_stack: list) -> None:
        self._task_stack = task_stack

    async def on_enter(self):
        while len(self._task_stack) > 0:
            self._current_task = self._task_stack.pop()
            self._current_agent_task = await self._current_task.create_new_task()
            # TODO add function tools after task creation
            current_tools = self._current_agent_task.tools
            current_tools.append(self._out_of_scope_func)
            await self._current_agent_task.update_tools(current_tools)

            result = await self._current_agent_task
            if result is not None:
                self._task_results[self._current_task] = result

                if self._current_task not in self._visited_tasks:
                    self._visited_tasks[id(self._current_task)] = self._current_task

        self.complete(self._task_results)

    def construct_out_of_scope_tool(self, *, task_id_lookup: dict) -> FunctionTool:
        task_ids = task_id_lookup.values()

        @function_tool
        async def out_of_scope(
            task_id: Annotated[
                int,
                Field(
                    description="The ID of the task requested",
                    json_schema_extra={"enum": list(task_ids)},
                ),
            ],
        ):
            if task_id in self._visited_tasks.keys():
                self._current_agent_task.complete(None)
                self._task_stack.append(self._current_task)
                self._task_stack.append(self._visited_tasks[task_id])
                return
            else:
                raise ToolError(
                    "Unable to regress, requested task not found in previously visited tasks"
                )

        return out_of_scope
