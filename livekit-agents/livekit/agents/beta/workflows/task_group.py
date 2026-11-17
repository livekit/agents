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
class TaskGroupProgress:
    """Where a TaskGroup stands, so a new TaskGroup can start from the same point.

    Read it from `TaskGroup.progress` and pass it to `TaskGroup(progress=...)`, for example
    when every inbound message of a text conversation is handled in a new session.

    Attributes:
        pending (list[str]): IDs of the tasks still to run, the current task first.
        results (dict[str, Any]): The result of each completed task, by ID.
        visited (list[str]): IDs of the tasks entered so far. The out_of_scope tool can
            regress to these.
    """

    pending: list[str]
    results: dict[str, Any]
    visited: list[str]


def _copy_progress(progress: TaskGroupProgress) -> TaskGroupProgress:
    return TaskGroupProgress(
        pending=list(progress.pending),
        results=dict(progress.results),
        visited=list(progress.visited),
    )


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
        progress: TaskGroupProgress | None = None,
        current_task: AgentTask | None = None,
    ):
        """TaskGroup orchestrates a sequence of multiple AgentTasks. It also allows for users to regress to previous tasks if requested.

        Args:
            summarize_chat_ctx (bool): Whether or not to summarize the interactions within the TaskGroup into one message and merge the context. Defaults to True.
            return_exceptions (bool): Whether or not to directly propagate an error. When set to True, the exception is added to the results dictionary and the sequence continues. Defaults to False.
            on_task_completed (Callable[]): A callable that executes upon each task completion. The callback takes in a single argument of a TaskCompletedEvent.
            progress (TaskGroupProgress | None): Saved progress to start from, as read from `TaskGroup.progress`, e.g. in an earlier session. The group runs `progress.pending` in order, keeps `progress.results` as the results of the tasks already completed, and lets the out_of_scope tool regress to `progress.visited`. Every ID must belong to a task added with `add()`, otherwise the group fails with a ValueError when it starts. Defaults to None: start from the first task.
            current_task (AgentTask | None): With `progress`, the task to run for `progress.pending[0]` instead of calling its factory. It keeps the chat context it was built with (e.g. its own conversation from the earlier session) rather than receiving a copy of the group's, and it can be built knowing that it is resuming. Later tasks and regressions use the factories. Defaults to None.
        """
        if current_task is not None and (progress is None or not progress.pending):
            raise ValueError("current_task needs a progress with a pending task to run it for")

        super().__init__(
            instructions="*empty*", chat_ctx=chat_ctx, llm=NOT_GIVEN
        )  # the LLM is set as NOT_GIVEN to allow session reusage if supported

        self._summarize_chat_ctx = summarize_chat_ctx
        self._return_exceptions = return_exceptions
        self._visited_tasks = set[str]()
        self._registered_factories: OrderedDict[str, _FactoryInfo] = OrderedDict()
        self._task_completed_callback = on_task_completed

        self._initial_progress = _copy_progress(progress) if progress is not None else None
        self._resumed_task = current_task
        self._started = False
        self._task_stack: list[str] = []
        self._task_results: dict[str, Any] = {}
        self._current_task: AgentTask | None = None
        self._current_task_id: str | None = None

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

    @property
    def progress(self) -> TaskGroupProgress:
        """A snapshot of the group's progress, to start a new TaskGroup from later.

        `pending` lists the running task first. Before the group starts, this is the
        progress it will start from. The results are the tasks' own objects, not copies;
        storing them (e.g. as JSON) is up to the caller.
        """
        if not self._started:
            if self._initial_progress is not None:
                return _copy_progress(self._initial_progress)
            return TaskGroupProgress(
                pending=list(self._registered_factories), results={}, visited=[]
            )

        pending = list(self._task_stack)
        if self._current_task_id is not None:
            pending.insert(0, self._current_task_id)
        return TaskGroupProgress(
            pending=pending,
            results=dict(self._task_results),
            # registration order, so the snapshot doesn't depend on set ordering
            visited=[
                task_id for task_id in self._registered_factories if task_id in self._visited_tasks
            ],
        )

    @property
    def current_task(self) -> AgentTask | None:
        """The task that is running, or the last one that ran. None before the group starts."""
        return self._current_task

    async def on_enter(self) -> None:
        progress = self._initial_progress
        if progress is not None:
            named = dict.fromkeys([*progress.pending, *progress.results, *progress.visited])
            unknown = [task_id for task_id in named if task_id not in self._registered_factories]
            if unknown:
                self.complete(ValueError(f"progress names unregistered task ids: {unknown}"))
                return
            self._task_stack = list(progress.pending)
            self._task_results = dict(progress.results)
            self._visited_tasks = set(progress.visited)
        else:
            self._task_stack = list(self._registered_factories.keys())
            self._task_results = {}
        self._started = True
        resumed_task = self._resumed_task

        while len(self._task_stack) > 0:
            task_id = self._task_stack.pop(0)
            factory_info = self._registered_factories[task_id]
            self._current_task_id = task_id

            if resumed_task is not None:
                # keep the context it was built with: it continues its own conversation
                task, resumed_task = resumed_task, None
            else:
                task = factory_info.task_factory()
                shared_chat_ctx = self.chat_ctx.copy()
                await task.update_chat_ctx(shared_chat_ctx)
            self._current_task = task

            if out_of_scope_tool := self._build_out_of_scope_tool(active_task_id=task_id):
                current_tools = task.tools
                current_tools.append(out_of_scope_tool)
                await task.update_tools(current_tools)

            try:
                self._visited_tasks.add(task_id)
                res = await task

                # AgentTask handoff merges omit function calls. Re-merge the completed
                # task context so task-group summarization can incorporate tool results.
                self._chat_ctx.merge(
                    task.chat_ctx.copy(),
                    exclude_instructions=True,
                )

                self._task_results[task_id] = res
                self._current_task_id = None

                if self._task_completed_callback is not None:
                    await self._task_completed_callback(
                        TaskCompletedEvent(agent_task=task, task_id=task_id, result=res)
                    )
            except _OutOfScopeError as e:
                self._current_task_id = None
                self._task_stack.insert(0, task_id)
                for task_id in reversed(e.target_task_ids):
                    self._task_stack.insert(0, task_id)
                continue
            except Exception as e:
                if self._return_exceptions:
                    self._task_results[task_id] = e
                    self._current_task_id = None
                    continue
                else:
                    self.complete(e)
                    return

        if self._summarize_chat_ctx:
            try:
                assert isinstance(self.session.llm, llm.LLM), (
                    "llm must be a LLM instance to summarize the chat_ctx"
                )

                # when a task is done, the chat_ctx is going to be merged with the "caller" chat_ctx
                # enabling summarization will result on only one ChatMessage added.
                # keep every item to allow summarization to be more action-aware.
                summarized_chat_ctx = await self.chat_ctx.copy(
                    exclude_instructions=False,
                    exclude_handoff=False,
                    exclude_config_update=False,
                    exclude_empty_message=False,
                    exclude_function_call=False,
                )._summarize(llm_v=self.session.llm, keep_last_turns=0)

                await self.update_chat_ctx(summarized_chat_ctx)
            except Exception as e:
                self.complete(e)
                return

        self.complete(TaskGroupResult(task_results=dict(self._task_results)))

    def _build_out_of_scope_tool(self, *, active_task_id: str) -> FunctionTool | None:
        # Only allow to regress to already visited tasks
        task_ids = self._visited_tasks.copy()
        task_ids.discard(active_task_id)
        if not task_ids:
            # nothing to regress to; providers reject an empty enum (e.g. Anthropic)
            return None

        task_repr = {
            f.id: f.description for f in self._registered_factories.values() if f.id in task_ids
        }

        description = (
            "Call to regress to other tasks according to what the user requested to modify, return the corresponding task ids. "
            'For example, if the user wants to change their email and there is a task with id "email_task" with a description of "Collect the user\'s email", return the id ("get_email_task").'
            "If the user requests to regress to multiple tasks, such as changing their phone number and email, return both task ids in the order they were requested."
            f"The following are the IDs and their corresponding task description. {json.dumps(task_repr, ensure_ascii=False)}"
        )

        @function_tool(description=description, flags=ToolFlag.IGNORE_ON_ENTER)
        async def out_of_scope(
            task_ids: Annotated[
                list[str],
                Field(
                    description="The IDs of the tasks requested",
                    json_schema_extra={"items": {"type": "string", "enum": list(task_ids)}},
                ),
            ],
        ) -> None:
            for task_id in task_ids:
                if task_id not in self._registered_factories or task_id not in self._visited_tasks:
                    raise ToolError(f"unable to regress, invalid task id {task_id}")

            if self._current_task is not None and not self._current_task.done():
                self._current_task.complete(_OutOfScopeError(target_task_ids=task_ids))

        return out_of_scope
