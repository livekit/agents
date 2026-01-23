"""
Azure Voice Live API Agent Demo

Demonstrates Azure Voice Live realtime voice agent with function calling capabilities.

Features:
- Azure Voice Live API for end-to-end speech-to-speech
- Server-side Voice Activity Detection (VAD)
- Function calling (weather, web search, jokes)
- Natural voice interaction
- Real-time audio streaming

Installation:
    pip install python-weather duckduckgo-search jokeapi

Environment Variables:
    AZURE_VOICELIVE_ENDPOINT - Azure Voice Live endpoint (wss://...)
    AZURE_VOICELIVE_API_KEY - Azure API key
    AZURE_VOICELIVE_MODEL - Azure model name (default: gpt-4o)
    LIVEKIT_URL - LiveKit server URL
    LIVEKIT_API_KEY - LiveKit API key
    LIVEKIT_API_SECRET - LiveKit API secret

Usage:
    python azure_voice_live_agent.py console

Try asking:
 - "What's the weather in Seattle?"
 - "Tell me a programming joke"
 - "Search for information about Azure AI"
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

import python_weather
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from jokeapi import Jokes

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    ToolError,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import azure

load_dotenv()

# Suppress verbose Azure SDK debug logs
logging.getLogger("azure").setLevel(logging.WARNING)

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
    Tell a joke that pertains to the category of the user's request.

    Args:
        category (str): The category of joke to tell.
            Available categories are: Any, Misc, Programming, Dark, Pun, Spooky, Christmas
    """
    try:
        category_map = {
            "puns": "Pun",
            "pun": "Pun",
            "any": "Any",
            "general": "Any",
            "misc": "Misc",
            "miscellaneous": "Misc",
            "programming": "Programming",
            "dark": "Dark",
            "spooky": "Spooky",
            "christmas": "Christmas",
        }
        normalized = category_map.get(category.lower(), "Any")

        j = await Jokes()
        joke = await j.get_joke(category=[normalized], safe_mode=True)
        if joke["type"] == "single":
            return {"joke": joke["joke"]}
        else:
            return {"setup": joke["setup"], "delivery": joke["delivery"]}
    except Exception as e:
        return ToolError(f"Error getting joke: {e}")


class AzureVoiceAssistant(Agent):
    def __init__(self, name: str = "Azure") -> None:
        current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        super().__init__(
            instructions=f"Your name is {name}, and you are a friendly voice assistant powered by Azure Voice Live API. "
            f"The current date and time is {current_time}. "
            "You can check the weather, search the web, and tell jokes. "
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
            "but you're also happy to have a friendly conversation."
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    print("🎙️ Using Azure Voice Live Realtime API with Semantic VAD")

    # Create the agent with instructions and tools
    agent = AzureVoiceAssistant(name="Azure")

    # Configure Semantic VAD for intelligent turn detection
    # Options:
    # - AzureSemanticVad: Default semantic VAD (multilingual)
    # - AzureSemanticVadEn: English-only, optimized for English
    # - AzureSemanticVadMultilingual: Explicit multilingual support
    from azure.ai.voicelive.models import AzureSemanticVadEn

    turn_detection = AzureSemanticVadEn(
        threshold=0.5,                    # Voice activity detection threshold (0.0-1.0)
        silence_duration_ms=500,          # Silence duration before turn ends
        prefix_padding_ms=300,            # Audio padding before speech
        speech_duration_ms=200,           # Minimum speech duration to trigger detection
        remove_filler_words=True,         # Remove filler words like "um", "uh"
    )

    # Create Azure Voice Live session
    session = AgentSession(
        llm=azure.realtime.RealtimeModel(
            endpoint=os.getenv("AZURE_VOICELIVE_ENDPOINT"),
            api_key=os.getenv("AZURE_VOICELIVE_API_KEY"),
            model=os.getenv("AZURE_VOICELIVE_MODEL", "gpt-4o"),
            voice=os.getenv("AZURE_VOICELIVE_VOICE", "en-US-AvaNeural"),
            turn_detection=turn_detection,  # Use semantic VAD
            tool_choice="auto",
        )
    )

    # Start the agent session
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                sample_rate=24000,
                num_channels=1,
            ),
            audio_output=room_io.AudioOutputOptions(
                sample_rate=24000,
                num_channels=1,
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
