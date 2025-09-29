import asyncio
from ...voice import Agent


class TaskOrchestrator(Agent):
    def __init__(self, *, llm):
        self._task_stack = []
        self._visited_tasks = []
        self._main_atask = None
        self._current_task = None
        # initialize a separate LLM to monitor and dispatch

    @property
    def task_stack(self) -> list:
        return self._task_stack
    
    @property
    def visited_tasks(self) -> list:
        return self._visited_tasks
    
    def start(self):
        self._main_atask = asyncio.create_task(self._main_task())

    def drain(self):
        ...

    async def _main_task(self):
        while len(self._task_stack) > 0:
            currtask = self._task_stack.pop()
            if currtask in self._visited_tasks:
                # TODO pass last user message as the trigger, retrieve new result
                ans = currtask.regress(trigger="")
            res = await currtask()
            self._visited_tasks.append(currtask)