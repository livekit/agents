import asyncio
from functools import partial

from livekit.agents import AgentSession, RunContext, ToolFlag, function_tool, Agent
from livekit.agents.voice import SpeechHandle
from livekit.agents.llm import FunctionCall
from livekit.durable import DurableScheduler, EffectCall, durable


async def my_network_call() -> None:
    return 6


@durable
async def nested_durable() -> None:
    await EffectCall(asyncio.sleep(5))


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a weather agent.")

    @function_tool(name="get_weather", flags=ToolFlag.DURABLE)
    async def my_function_tool(self, ctx: RunContext, location: str) -> str:
        """my_function_tool

        Args:
            location: the location to get the weather for
        """
        print("self", self)
        print("my_function_tool", ctx, location)
        result = await EffectCall(my_network_call())
        print("a", result)

        await EffectCall(asyncio.sleep(5))

        await nested_durable()

        e = await EffectCall(asyncio.sleep(5))
        print("b", e)
        print("self", self)
        # await MyAgentTask()

        return "task done"


async def amain() -> None:
    session = AgentSession()
    ctx = RunContext(
        session=session,
        speech_handle=SpeechHandle(speech_id="speech_123", allow_interruptions=True),
        function_call=FunctionCall(
            call_id="function_call_123",
            name="my_function_tool",
            arguments="{}",
        ),
    )

    agent = MyAgent()

    scheduler = DurableScheduler()
    print(agent.my_function_tool.info)
    scheduler.execute(partial(agent.my_function_tool, ctx, "New York"))

    await asyncio.sleep(5)
    print("dumping scheduler")
    states = await scheduler.checkpoint()
    await scheduler.aclose()
    print("scheduler closed")

    scheduler = DurableScheduler()
    print("loading scheduler")
    results = await asyncio.gather(*scheduler.restore(states))
    print("results", results)


asyncio.run(amain())
