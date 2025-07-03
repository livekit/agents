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
"""

import asyncio
import random
from typing import Any

from dotenv import load_dotenv
from jokeapi import Jokes

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    RoomInputOptions,
    RoomOutputOptions,
    ToolError,
    llm,
)
from livekit.agents.llm import function_tool
from livekit.agents.llm.chat_context import ChatContext
from livekit.plugins import aws

load_dotenv()


weather_schema = {
    "name": "get_weather",
    "description": "Retrieve the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "The city to get the weather for"},
            "units": {
                "type": "string",
                "description": "The units to use for the temperature in degrees (Celsius or Fahrenheit)", # noqa: E501
                "default": "fahrenheit",
                "enum": ["celsius", "fahrenheit"],
            },
        },
        "required": ["city"],
    },
}


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
@function_tool
async def search_web(query: str) -> dict[str, Any]:
    """
    Search the web for information about a given query.

    Args:
        query (str): The query to search for.
    """
    return ToolError("No results found")


@function_tool
async def tell_joke(category: list[str] = None) -> dict[str, Any]:
    """
    Tell a joke that pertains to the category of the user's request.

    Args:
        category (list[str]): The category of joke to tell.
    """
    j = await Jokes()
    joke = await j.get_joke(category=category)
    if joke["type"] == "single":
        return {"joke": joke["joke"]}
    else:
        return {"setup": joke["setup"], "delivery": joke["delivery"]}


story = """
There was once an old man and old woman who were not blessed with little ones.
The home seemed bare, with no children to skitter through the kitchen,
make pillow castles, or laugh in the four corners of the cottage.
The garden felt wintry, with no bright legs dangling from the branches of the trees.

"Old man", said the old woman.
"I yearn for a son. There is a log in the yard. Will you carve it into a boy for us?"
The old man carved the log into the shape of a boy. It was strong, brown, and solidly built.
The old woman wept with happiness when she saw it. She kissed the tip of its nose and
placed it in the same cradle she herself had slept in as a child.
The next day, the log was gone. In its place was a soft, curled, sleeping little human.
It had turned into a real boy.
His mother called him Ivaysk-- Little Stick.
"""


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
                    max_tokens=2048,
                )
            )
            await session.start(
                room=ctx.room,
                agent=Assistant(tools=[get_weather, get_median_home_price, search_web, tell_joke]),
                room_input_options=RoomInputOptions(close_on_disconnect=False),
                room_output_options=RoomOutputOptions(
                    audio_enabled=True,
                    audio_sample_rate=24000,
                    audio_num_channels=1,
                    transcription_enabled=True,
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
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
