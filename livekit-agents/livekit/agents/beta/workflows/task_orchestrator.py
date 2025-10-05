import json
from typing import Annotated

from pydantic import Field

from ... import FunctionTool, llm
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
            instructions="""You are a task orchestrator managing a defined workflow. There is a stack of tasks, and if the user wants to regress to a previous question, call out_of_scope().
            """,
            chat_ctx=chat_ctx,
            llm=llm,
        )
        self._task_stack = task_stack
        self._task_order = task_stack[::-1]
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
    def task_stack(self) -> list:
        return self._task_stack

    @property
    def visited_tasks(self) -> dict:
        return self._visited_tasks

    @task_stack.setter
    def task_stack(self, task_stack: list) -> None:
        self._task_stack = task_stack

    def drain(self): ...  # or rename to aclose

    async def on_enter(self):
        while len(self._task_stack) > 0:
            self._current_task = self._task_stack.pop()
            self._current_agent_task = self._current_task.task
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
