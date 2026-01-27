import asyncio
from functools import partial

from livekit.agents import ToolFlag, function_tool
from livekit.durable import DurableScheduler, EffectCall, durable


async def my_network_call() -> None:
    return 6


@durable
async def nested_durable() -> None:
    await EffectCall(asyncio.sleep(5))


@function_tool(name="get_weather", flags=ToolFlag.DURABLE)
async def my_function_tool(location: str) -> str:
    """my_function_tool

    Args:
        location: the location to get the weather for
    """
    result = await EffectCall(my_network_call())
    print("a", result)

    await EffectCall(asyncio.sleep(5))

    await nested_durable()

    e = await EffectCall(asyncio.sleep(5))
    print("b", e)
    # await MyAgentTask()


async def amain() -> None:
    scheduler = DurableScheduler()
    print(my_function_tool.info)
    scheduler.execute(partial(my_function_tool, "New York"))

    await asyncio.sleep(5)
    print("dumping scheduler")
    states = await scheduler.checkpoint()
    await scheduler.aclose()
    print("scheduler closed")

    scheduler = DurableScheduler()
    print("loading scheduler")
    await asyncio.gather(*scheduler.restore(states))


asyncio.run(amain())
