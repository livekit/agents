import json
import sys
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from typing import Annotated, Any

from pydantic import Field

from ... import FunctionTool, llm
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask


@dataclass
class _FactoryInfo:
    task_factory: Callable[[], AgentTask]
    id: str
    description: str


@dataclass
class TaskGroupResult:
    task_results: dict[str, Any]


@dataclass
class TaskCompletedEvent:
    agent_task: AgentTask
    task_id: str
    result: Any


class _OutOfScopeError(ToolError):
    def __init__(self, target_task_ids: list) -> None:
        self.target_task_ids = target_task_ids


class TaskGroup(AgentTask[TaskGroupResult]):
    def __init__(
        self,
        *,
        summarize_chat_ctx: bool = True,
        return_exceptions: bool = False,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        on_task_completed: Callable[[TaskCompletedEvent], Coroutine[None, None, None]]
        | None = None,
    ):
        """TaskGroup orchestrates a sequence of multiple AgentTasks. It also allows for users to regress to previous tasks if requested.

        Args:
            summarize_chat_ctx (bool): Whether or not to summarize the interactions within the TaskGroup into one message and merge the context. Defaults to True.
            return_exceptions (bool): Whether or not to directly propagate an error. When set to True, the exception is added to the results dictionary and the sequence continues. Defaults to False.
            on_task_completed (Callable[]): A callable that executes upon each task completion. The callback takes in a single argument of a TaskCompletedEvent.
        """
        super().__init__(instructions="*empty*", chat_ctx=chat_ctx, llm=None)

        self._summarize_chat_ctx = summarize_chat_ctx
        self._return_exceptions = return_exceptions
        self._visited_tasks = set[str]()
        self._registered_factories: OrderedDict[str, _FactoryInfo] = OrderedDict()
        self._task_completed_callback = on_task_completed

    def add(
        self,
        task_factory: Callable[[], AgentTask],
        *,
        id: str,
        description: str,
    ) -> Self:
        """Adds an AgentTask to the TaskGroup.

        Args:
            task_factory (Callable): A callable that returns a task instance
            id (str): An identifier for the task used to access results
            description (str): A description that helps the LLM understand when to regress to this task
        """
        self._registered_factories[id] = _FactoryInfo(
            task_factory=task_factory, id=id, description=description
        )
        return self

    async def on_enter(self) -> None:
        task_stack = list(self._registered_factories.keys())
        task_results: dict[str, Any] = {}

        while len(task_stack) > 0:
            task_id = task_stack.pop(0)
            factory_info = self._registered_factories[task_id]

            self._current_task = factory_info.task_factory()

            shared_chat_ctx = self.chat_ctx.copy()
            await self._current_task.update_chat_ctx(shared_chat_ctx)

            if out_of_scope_tool := self._build_out_of_scope_tool(active_task_id=task_id):
                current_tools = self._current_task.tools
                current_tools.append(out_of_scope_tool)
                await self._current_task.update_tools(current_tools)

            try:
                self._visited_tasks.add(task_id)
                res = await self._current_task
                task_results[task_id] = res
                if self._task_completed_callback is not None:
                    await self._task_completed_callback(
                        TaskCompletedEvent(
                            agent_task=self._current_task, task_id=task_id, result=res
                        )
                    )
            except _OutOfScopeError as e:
                task_stack.insert(0, task_id)
                for task_id in reversed(e.target_task_ids):
                    task_stack.insert(0, task_id)
                continue
            except Exception as e:
                if self._return_exceptions:
                    task_results[task_id] = e
                    continue
                else:
                    self.complete(e)
                    return

        try:
            if self._summarize_chat_ctx:
                assert isinstance(self.session.llm, llm.LLM)

                # when a task is done, the chat_ctx is going to be merged with the "caller" chat_ctx
                # enabling summarization will result on only one ChatMessage added.
                summarized_chat_ctx = await self.chat_ctx.copy(
                    exclude_instructions=True,
                    exclude_handoff=True,
                    exclude_config_update=True,
                    exclude_empty_message=True,
                    exclude_function_call=True,
                )._summarize(llm_v=self.session.llm, keep_last_turns=0)
                await self.update_chat_ctx(summarized_chat_ctx)
        except Exception as e:
            self.complete(RuntimeError(f"failed to summarize the chat_ctx: {e}"))
        self.complete(TaskGroupResult(task_results=task_results))

    def _build_out_of_scope_tool(self, *, active_task_id: str) -> FunctionTool | None:
        if not self._visited_tasks:
            return None

        # Only allow to regress to already visited tasks
        task_ids = self._visited_tasks.copy()
        task_ids.discard(active_task_id)
        task_repr = {
            f.id: f.description for f in self._registered_factories.values() if f.id in task_ids
        }

        description = (
            "Call to regress to other tasks according to what the user requested to modify, return the corresponding task ids. "
            'For example, if the user wants to change their email and there is a task with id "email_task" with a description of "Collect the user\'s email", return the id ("get_email_task").'
            "If the user requests to regress to multiple tasks, such as changing their phone number and email, return both task ids in the order they were requested."
            f"The following are the IDs and their corresponding task description. {json.dumps(task_repr)}"
        )

        @function_tool(description=description, flags=ToolFlag.IGNORE_ON_ENTER)
        async def out_of_scope(
            task_ids: Annotated[
                list[str],
                Field(
                    description="The IDs of the tasks requested",
                    json_schema_extra={"items": {"enum": list(task_ids)}},
                ),
            ],
        ) -> None:
            for task_id in task_ids:
                if task_id not in self._registered_factories or task_id not in self._visited_tasks:
                    raise ToolError(f"unable to regress, invalid task id {task_id}")

            if not self._current_task.done():
                self._current_task.complete(_OutOfScopeError(target_task_ids=task_ids))

        return out_of_scope
