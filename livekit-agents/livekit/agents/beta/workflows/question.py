from dataclasses import dataclass
from typing import Literal

from ... import llm
from ...llm.tool_context import function_tool
from ...voice.agent import AgentTask
from ...voice.events import RunContext

# TODO: restrict function tool answer type to defined return_type


@dataclass
class Question(AgentTask):
    def __init__(
        self,
        return_type: Literal["bool", "float", "str"] = "str",
        instructions: str | None = None,
        prompt: str | None = None,
        mutable: bool = True,
        chat_ctx: llm.ChatContext | None = None,
    ):
        """Create a new instance of Question. Intended for simple information collection with little to no
        input validation.

        Args:
            return_type (Literal["bool", "float", "str"]): The expected return type of the question. -> WIP
            instructions (str): Passed when you want to use generate_reply() to ask the question or if you need more control
            prompt (str): Passed when you want to use say() to ask the question -> WIP
            mutable (bool): True if the answer is able to be changed, false if not
        You are able to pass both instructions and a prompt, but at least one must be passed.
        """
        if instructions is None and prompt is None:
            raise Exception("Either 'instructions' or 'prompt' parameter must be given")

        super().__init__(
            instructions=instructions,
        )
        self._return_type = return_type
        self._prompt = prompt
        self._instructions = instructions
        self._mutable = mutable
        # self._chat_ctx = chat_ctx

        self._paused = False
        self._result = None
        self._soft_complete_future = None

    @property
    def paused(self) -> bool:
        """Defines if the Question is in a paused state"""
        return self._paused

    @property
    def mutable(self) -> bool:
        """Defines if the Question is mutable."""
        return self._mutable

    @property
    def result(self) -> bool | str | float | None:
        """Returns the collected result of the question"""
        return self._result

    @paused.setter
    def pause(self, state: bool) -> None:
        self._paused = state

    @mutable.setter
    def mutable(self, state: bool) -> None:
        self._mutable = state

    async def on_enter(self) -> None:
        """Runs on the first encounter of the question"""
        if self._prompt:
            await self.session.say(text=self._prompt)
        else:
            await self.session.generate_reply()

    async def regress(self, trigger: str | None = None):
        """Call to return to this question. If the question has been answered already, return a new instance with copied context.

        Args:
            trigger (str): The last thing the user said to trigger the regression to this question
            - only applicable when this question has a result already
        """
        if self._result:
            self._chat_ctx.add_message(role="user", content=trigger)
            return Question(
                return_type=self._return_type,
                instructions=self._instructions,
                prompt=self._prompt,
                chat_ctx=self._chat_ctx,
            )
        else:
            # assume that the question has NOT been answered and the user regressed to another question before responding.
            # thus, we will ask the question again and collect the answer.
            if self._prompt:
                await self.session.say(text=self._prompt)
            else:
                await self.session.generate_reply()

    @function_tool()
    async def answer_received(self, context: RunContext, user_answer: str | bool | float):
        """Call this when the user provides an answer to the question asked"""
        self._result = user_answer

        self.complete(user_answer)
