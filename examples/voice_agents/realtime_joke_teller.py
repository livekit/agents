"""
This is an example voice agent that uses Amazon Nova Sonic and showcases how to:
 - set system prompt
 - pass in a ChatContext to the agent
 - set tool_choice
 - configure the inference parameters of the model
 - register tools

There is an event loop that waits for a participant to join the room
and then starts the agent session. The agent session is closed when the participant leaves.

Try asking the agent to tell a joke about a specific category!

Note: install the required dependencies
```
uv pip install jokeapi ddgs
```
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from ddgs import DDGS
from dotenv import load_dotenv
from jokeapi import Jokes

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    RunContext,
    ToolError,
    llm,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.agents.llm.chat_context import ChatContext
from livekit.plugins import aws

load_dotenv()

g = DDGS()

weather_schema = {
    "name": "get_weather",
    "description": "Retrieve the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city to get the weather for"},
            "units": {
                "type": "string",
                "description": "The units to use for the temperature in degrees (Celsius or Fahrenheit)",  # noqa: E501
                "default": "fahrenheit",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "required": ["city"],
    },
}


@dataclass
class MySessionInfo:
    user_name: str | None = None
    age: int | None = None


# example of how to create a RawFunctionTool
@function_tool(raw_schema=weather_schema)
async def get_weather(raw_arguments: dict[str, Any]) -> dict[str, Any]:
    city = raw_arguments["city"]
    units = raw_arguments.get("units", "fahrenheit")
    if units == "celsius":
        temp = random.randint(0, 35)
    else:
        temp = random.randint(32, 100)
    return {"temperature": temp, "units": units, "city": city}


# example of how to create a FunctionTool
# note that if raw_schema is absent, you should be providing a docstring
@function_tool
async def get_median_home_price(location: str) -> dict[str, Any]:
    """
    Get the median home price for a given location.

    Args:
        location (str): The location to get the median home price for.

    Returns:
        dict[str, Any]: A dictionary containing the median home price and the location.
    """
    price = random.randint(100000, 1000000)
    return {"median_home_price": f"${price:,.0f}", "location": location}


# example of how to handle a tool call that returns a ToolError
# note: if you want the model to gracefully handle the error, return a dict with an "error" key
@function_tool
async def search_web(query: str, max_results: int = 1) -> dict[str, Any]:
    """
    Search the web using DuckDuckGo search engine for information about a given query.

    Args:
        query (str): The query to search for.
        max_results (int): The maximum number of results to return.

    Returns:
        dict[str, Any]: A dictionary containing the search results.
        The keys are the index of the result and the values are another dictionary with the following keys:
        - title: Title of the result.
        - url: URL of the result.
        - body: Body of the result.
    """  # noqa: E501
    try:
        results = g.text(query, max_results=max_results)
    except Exception as e:
        return ToolError(f"Error searching the web: {e}")
    d = {str(i): res for i, res in enumerate(results)}
    for v in d.values():
        v["url"] = v.pop("href")
    return d


@function_tool
async def tell_joke(category: list[str] | None = None) -> dict[str, Any]:
    """
    Tell a joke that pertains to the category of the user's request.

    Args:
        category (list[str]): The category of joke to tell.
            Available categories are: Any, Misc, Programming, Dark, Pun, Spooky, Christmas
    """
    j = await Jokes()
    joke = await j.get_joke(category=category if category is not None else ["Any"])
    if joke["type"] == "single":
        return {"joke": joke["joke"]}
    else:
        return {"setup": joke["setup"], "delivery": joke["delivery"]}


story = """
High up in the sky lived a small, fluffy cloud named Wispy.
Unlike the other clouds that loved to bunch together, Wispy enjoyed floating alone and watching the world below.
One day, he noticed a tiny flower in a garden that looked very thirsty.
"Oh dear," thought Wispy, "that poor flower needs some water!"
Wispy tried his best to make himself rain, but he was too small to make more than a few drops.
The other clouds saw him struggling and felt sorry for him.
"Need some help?" asked a big, friendly cloud named Thunder.
Soon, all the clouds came together around Wispy, and together they made a gentle rain shower that gave the flower exactly what it needed.
The flower perked up and bloomed beautifully, showing off its bright pink petals
From that day on, Wispy learned that while it's nice to be independent, working together with friends can help accomplish wonderful things.
He still loved floating alone sometimes, but he always knew his cloud friends were there when he needed them.
"""  # noqa: E501


class Assistant(Agent):
    def __init__(self, tools: list[llm.FunctionTool | llm.RawFunctionTool]) -> None:
        # example of how to pass in ChatContext
        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(role="user", content="hey sonic, tell me a children's story")
        chat_ctx.add_message(role="assistant", content=story)

        super().__init__(
            instructions="You are a helpful voice AI assistant.",
            tools=tools,
            chat_ctx=chat_ctx,
        )

    # example of how to use the RunContext to fetch userdata
    @function_tool
    async def get_user_name_and_age(self, context: RunContext[MySessionInfo]) -> dict[str, Any]:
        """
        Get the user name and age for the current session.
        """
        return {"user_name": context.userdata.user_name, "age": context.userdata.age}


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    session: AgentSession | None = None
    try:
        # Connect to the LiveKit server
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # wait for either a participant to join or a shutdown signal
        shutdown_future: asyncio.Future[None] = asyncio.Future()

        async def _on_shutdown(_reason: str) -> None:
            if not shutdown_future.done():
                shutdown_future.set_result(None)

        ctx.add_shutdown_callback(_on_shutdown)

        while True:
            participant_task = asyncio.create_task(
                ctx.wait_for_participant(
                    kind=[
                        rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD,
                    ]
                )
            )
            done, _ = await asyncio.wait(
                [participant_task, shutdown_future], return_when=asyncio.FIRST_COMPLETED
            )

            if shutdown_future in done:
                break

            session = AgentSession(
                llm=aws.realtime.RealtimeModel(
                    # example of how to set tool_choice
                    tool_choice="auto",
                    max_tokens=10_000,
                )
            )
            await session.start(
                room=ctx.room,
                agent=Assistant(tools=[get_weather, get_median_home_price, search_web, tell_joke]),
                room_options=room_io.RoomOptions(
                    audio_input=room_io.AudioInputOptions(
                        sample_rate=24000,
                        num_channels=1,
                    ),
                    close_on_disconnect=False,
                ),
            )

            # wait for either last participant to leave or a shutdown signal
            room_empty_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()

            def _on_participant_disconnected(_: rtc.Participant, fut=room_empty_future) -> None:
                if len(ctx.room.remote_participants) == 0 and not fut.done():
                    fut.set_result(None)

            ctx.room.on("participant_disconnected", _on_participant_disconnected)

            try:
                # blocking wait for either future to be set
                await asyncio.wait(
                    [shutdown_future, room_empty_future], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                ctx.room.off("participant_disconnected", _on_participant_disconnected)
                await session.aclose()

                # reset session to None to avoid double-closing
                session = None
            if shutdown_future.done():
                break

    finally:
        # final cleanup in case of early exit
        if session:
            await session.aclose()


if __name__ == "__main__":
    agents.cli.run_app(server)
