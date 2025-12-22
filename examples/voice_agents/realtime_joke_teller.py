"""
Amazon Nova Sonic 2.0 Voice Agent Demo

Showcases AWS voice agent capabilities with two modes:

REALTIME MODE (default):
 - Amazon Nova Sonic 2.0 for end-to-end speech-to-speech
 - Lower latency, natural conversation flow
 - Single model handles everything

PIPELINE MODE:
 - AWS Transcribe (STT) + Bedrock LLM + Polly (TTS)
 - More control over each component
 - Mix and match different models

Both modes support:
 - Function calling (weather, web search, jokes)
 - Natural voice interaction
 - Tool execution

Installation:
    pip install python-weather duckduckgo-search jokeapi

Usage:
    # Realtime mode (default)
    python realtime_joke_teller.py console

    # Pipeline mode
    python realtime_joke_teller.py console --mode pipeline

Try asking:
 - "What's the weather in Seattle?"
 - "Tell me a programming joke"
 - "Search for information about AWS"
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any

import python_weather
from ddgs import DDGS
from dotenv import load_dotenv
from jokeapi import Jokes

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    ToolError,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import aws, silero

load_dotenv()

g = DDGS()


@function_tool
async def get_weather(city: str, units: str = "fahrenheit") -> dict[str, int | str] | ToolError:
    """
    Retrieve the current weather for a city.

    Args:
        city (str): The city to get the weather for.
        units (str): Temperature units - 'celsius' or 'fahrenheit' (default: fahrenheit).

    Returns:
        dict[str, Any]: Weather information including temperature, description, and city.
    """
    try:
        async with python_weather.Client(
            unit=python_weather.METRIC if units.lower() == "celsius" else python_weather.IMPERIAL
        ) as client:
            weather = await client.get(city)

            return {
                "temperature": weather.temperature,
                "units": units,
                "city": city,
                "description": weather.description,
                "feels_like": weather.feels_like,
                "humidity": weather.humidity,
            }
    except Exception as e:
        return ToolError(f"Error getting weather for {city}: {e}")


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
    """
    try:
        results = g.text(query, max_results=max_results)
    except Exception as e:
        return ToolError(f"Error searching the web: {e}")
    d = {str(i): res for i, res in enumerate(results)}
    for v in d.values():
        v["url"] = v.pop("href")
    return d


@function_tool
async def tell_joke(category: str = "Any") -> dict[str, Any] | ToolError:
    """
    Tell a joke that pertains to the category of the user's request. Just choose a Pun category if they don't specify

    Args:
        category (str): The category of joke to tell.
            Available categories are: Any, Misc, Programming, Dark, Pun, Spooky, Christmas
    """
    try:
        # Normalize category: capitalize first letter, handle variations
        category_map = {
            "puns": "Pun",
            "pun": "Pun",
            "any": "Any",
            "general": "Any",  # Map "general" to "Any"
            "misc": "Misc",
            "miscellaneous": "Misc",
            "programming": "Programming",
            "dark": "Dark",
            "spooky": "Spooky",
            "christmas": "Christmas",
        }
        # Default to "Any" if category not recognized
        normalized = category_map.get(category.lower(), "Any")

        j = await Jokes()
        joke = await j.get_joke(category=[normalized], safe_mode=True)
        if joke["type"] == "single":
            return {"joke": joke["joke"]}
        else:
            return {"setup": joke["setup"], "delivery": joke["delivery"]}
    except Exception as e:
        return ToolError(f"Error getting joke: {e}")


class Assistant(Agent):
    def __init__(self, name: str = "Sonic") -> None:
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        super().__init__(
            instructions=f"Your name is {name}, and you are a friendly and enthusiastic voice assistant. "
            f"The current date and time is {current_time}, just for your reference, no need to volunteer it to the user"
            "You love helping people and having natural conversations. "
            "You can check the weather anywhere in the world, search the web for information, and tell jokes. "
            "When telling jokes, always check if they're appropriate for all audiences before sharing. "
            "Avoid jokes with cultural humor, dark humor, offensive content, or adult themes. "
            "if you get a joke from the tool that you feel you shouldn't share, just call the tool again"
            "But you're also happy to just chat about anything! "
            "Be warm, conversational, and engaging. "
            "Keep your responses natural and concise for voice interaction. "
            "Always respond in the same language the user speaks to you.",
            tools=[get_weather, search_web, tell_joke],
        )
        self._name = name

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=f"Introduce yourself as {self._name} and greet the user warmly. "
            "Let them know you can help with weather, web searches, and jokes, "
            "but you're also happy to just have a friendly conversation about anything they'd like to talk about."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    session: AgentSession | None = None
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        shutdown_future: asyncio.Future[None] = asyncio.Future()

        async def _on_shutdown(_reason: str) -> None:
            if not shutdown_future.done():
                shutdown_future.set_result(None)

        ctx.add_shutdown_callback(_on_shutdown)

        while True:
            participant_task = asyncio.create_task(
                ctx.wait_for_participant(kind=[rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD])
            )
            done, _ = await asyncio.wait(
                [participant_task, shutdown_future], return_when=asyncio.FIRST_COMPLETED
            )

            if shutdown_future in done:
                break

            # Get mode from environment variable (set by CLI flag)
            agent_mode = os.getenv("AGENT_MODE", "realtime").lower()

            # Configure session based on mode
            if agent_mode == "pipeline":
                print("🔧 Using PIPELINE mode: STT + LLM + TTS")
                agent_name = "Nova"
                session = AgentSession(
                    stt=aws.STT(),
                    llm=aws.LLM(),
                    tts=aws.TTS(),
                    vad=silero.VAD.load(),
                )
            else:
                print("⚡ Using REALTIME mode: Nova Sonic 2.0")
                agent_name = "Sonic"
                session = AgentSession(
                    llm=aws.realtime.RealtimeModel.with_nova_sonic_2(
                        voice="tiffany",
                        tool_choice="auto",
                        max_tokens=10_000,
                    )
                )

            await session.start(
                room=ctx.room,
                agent=Assistant(name=agent_name),
                room_options=room_io.RoomOptions(
                    audio_input=room_io.AudioInputOptions(
                        sample_rate=24000,
                        num_channels=1,
                    ),
                    close_on_disconnect=False,
                ),
            )

            room_empty_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()

            def _on_participant_disconnected(_: rtc.Participant, fut=room_empty_future) -> None:
                if len(ctx.room.remote_participants) == 0 and not fut.done():
                    fut.set_result(None)

            ctx.room.on("participant_disconnected", _on_participant_disconnected)

            try:
                await asyncio.wait(
                    [shutdown_future, room_empty_future], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                ctx.room.off("participant_disconnected", _on_participant_disconnected)
                await session.aclose()
                session = None

            if shutdown_future.done():
                break

    finally:
        if session:
            await session.aclose()


if __name__ == "__main__":
    import sys

    # Parse --mode flag before LiveKit CLI processes args
    mode = "realtime"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
            # Remove --mode and its value from argv so LiveKit CLI doesn't see it
            sys.argv.pop(idx)  # Remove --mode
            sys.argv.pop(idx)  # Remove the value

    # Validate mode
    if mode not in ["realtime", "pipeline"]:
        print(f"Error: Invalid mode '{mode}'. Must be 'realtime' or 'pipeline'")
        sys.exit(1)

    # Set mode via environment variable
    os.environ["AGENT_MODE"] = mode

    # Run the agent
    agents.cli.run_app(server)
