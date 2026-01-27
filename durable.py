import asyncio

from livekit import durable
from livekit.agents.durable_scheduler import DurableScheduler, EffectCall


async def my_network_call() -> None:
    return 6


@durable.durable
async def nested_durable() -> None:
    await EffectCall(asyncio.sleep(5))


@durable.durable
async def my_function_tool() -> str:
    result = await EffectCall(my_network_call())
    print("a", result)

    await EffectCall(asyncio.sleep(5))

    await nested_durable()

    e = await EffectCall(asyncio.sleep(5))
    print("b", e)
    # await MyAgentTask()


async def amain() -> None:
    scheduler = DurableScheduler()
    scheduler.execute(my_function_tool)

    await asyncio.sleep(5)
    print("dumping scheduler")
    states = await scheduler.checkpoint()
    await scheduler.aclose()
    print("scheduler closed")

    scheduler = DurableScheduler()
    print("loading scheduler")
    await asyncio.gather(*scheduler.restore(states))


asyncio.run(amain())
