import random
from dataclasses import dataclass

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool, inference, llm


def _llm_model() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


class KellyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
        )

    @function_tool
    async def lookup_weather(self, ctx: RunContext, location: str):
        """Called when the user asks for weather related information.
        Args:
            location: The location they are asking for
        """
        return "sunny with a temperature of 70 degrees."

    @function_tool
    async def talk_to_echo(self, ctx: RunContext):
        """Called when the user wants to speak with Echo"""
        await self.session.say("Hello world")
        return EchoAgent()


class EchoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Your name is Echo. You would interact with users via voice.")

    async def on_enter(self) -> None:
        # AgentSession.run will even capture this generate_reply!
        self.session.generate_reply(user_input="Say hello to the user!")


@pytest.mark.asyncio
async def test_function_call():
    async with _llm_model() as llm, AgentSession(llm=llm) as sess:
        await sess.start(KellyAgent())

        result = await sess.run(user_input="What is the weather in San Francisco?")
        result.expect.next_event().is_function_call(
            name="lookup_weather", arguments={"location": "San Francisco"}
        )
        result.expect.next_event().is_function_call_output(
            output="sunny with a temperature of 70 degrees."
        )
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()

        result = await sess.run(user_input="Can I speak to Echo?")
        result.expect.next_event().is_function_call()
        result.expect.next_event().is_message(role="assistant")  # say `Hello world`!
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_agent_handoff(new_agent_type=EchoAgent)
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


@pytest.mark.asyncio
async def test_start_with_capture_run():
    async with _llm_model() as llm, AgentSession(llm=llm) as sess:
        result = await sess.start(EchoAgent(), capture_run=True)

        print(result.events)
        result.expect.next_event().is_agent_handoff(new_agent_type=EchoAgent)
        result.expect.next_event().is_message(role="assistant")

        result = await sess.run(user_input="Hello how are you?")
        print(result.events)
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()


@dataclass
class RandomResult:
    random_number: int


class InlineAgent(AgentTask[RandomResult]):
    def __init__(self, *, oneshot: bool) -> None:
        super().__init__(instructions="You are a voice assistant")
        self._oneshot = oneshot

    async def on_enter(self) -> None:
        if self._oneshot:
            self.session.generate_reply(instructions="Call the generate_number tool")

    @function_tool
    async def generate_number(self, ctx: RunContext):
        self.complete(RandomResult(random_number=random.randint(1, 100)))
        return None


class AshAgent(Agent):
    def __init__(self, *, oneshot: bool) -> None:
        super().__init__(instructions="Your name is Ash. You would interact with users via voice.")
        self._oneshot = oneshot

    @function_tool
    async def start_random_generator(self, ctx: RunContext):
        """Get the email address of the user"""
        random_result = await InlineAgent(oneshot=self._oneshot)
        return random_result.random_number


@pytest.mark.asyncio
async def test_inline_agent():
    async with _llm_model() as llm, AgentSession(llm=llm) as sess:
        await sess.start(AshAgent(oneshot=True))

        result = await sess.run(user_input="Start the random generator?")

    async with _llm_model() as llm, AgentSession(llm=llm) as sess:
        await sess.start(AshAgent(oneshot=False))

        result = await sess.run(user_input="Start the random generator?")
        print(result.events)
        result = await sess.run(user_input="Give me a random number?")
        print(result.events)
