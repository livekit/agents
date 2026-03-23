"""Example: ToolSearchToolset and ToolProxyToolset for dynamic tool discovery.

Both toolsets wrap a collection of tools and expose a `tool_search` function so the
LLM can discover tools on demand instead of loading them all upfront.

ToolSearchToolset:
  - Matched tools are added directly to the LLM's tool list on the next turn.
  - The model uses native tool calls to invoke discovered tools.
  - May be simpler for the model to understand.

ToolProxyToolset:
  - Exposes exactly two fixed tools: `tool_search` and `call_tool`.
  - The tool list never changes, so providers may reuse their prompt cache across turns.
  - May be better for many tools or cost-sensitive workloads.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    cli,
    inference,
    llm,
    metrics,
)
from livekit.agents.beta.toolsets import ToolProxyToolset, ToolSearchToolset
from livekit.agents.metrics.base import LLMMetrics
from livekit.plugins import silero

logger = logging.getLogger("tool-search-example")
logger.setLevel(logging.INFO)

load_dotenv()


class WeatherToolset(llm.Toolset):
    @llm.function_tool
    async def get_weather(self, location: str) -> str:
        """Get current weather for a location.

        Args:
            location: City name or region
        """
        logger.info(f"Getting weather for {location}")
        return f"Sunny, 72F in {location}"

    @llm.function_tool
    async def get_forecast(self, location: str, days: int) -> str:
        """Get weather forecast for upcoming days.

        Args:
            location: City name or region
            days: Number of days to forecast
        """
        logger.info(f"Getting {days}-day forecast for {location}")
        return f"{days}-day forecast for {location}: mostly sunny"


class FlightToolset(llm.Toolset):
    @llm.function_tool
    async def search_flights(self, origin: str, destination: str, date: str) -> str:
        """Search for available flights.

        Args:
            origin: Departure city or airport code
            destination: Arrival city or airport code
            date: Travel date
        """
        logger.info(f"Searching flights {origin} -> {destination} on {date}")
        return f"Found 3 flights from {origin} to {destination} on {date}"

    @llm.function_tool
    async def book_flight(self, flight_id: str) -> str:
        """Book a specific flight.

        Args:
            flight_id: The flight identifier to book
        """
        logger.info(f"Booking flight {flight_id}")
        return f"Flight {flight_id} booked successfully"


class HotelToolset(llm.Toolset):
    @llm.function_tool
    async def search_hotels(self, city: str, check_in: str, check_out: str) -> str:
        """Search for hotels in a city.

        Args:
            city: City to search hotels in
            check_in: Check-in date
            check_out: Check-out date
        """
        logger.info(f"Searching hotels in {city}")
        return f"Found 5 hotels in {city} from {check_in} to {check_out}"

    @llm.function_tool
    async def book_hotel(self, hotel_id: str) -> str:
        """Book a specific hotel.

        Args:
            hotel_id: The hotel identifier to book
        """
        logger.info(f"Booking hotel {hotel_id}")
        return f"Hotel {hotel_id} booked successfully"


@llm.function_tool
async def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert between currencies.

    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g. USD)
        to_currency: Target currency code (e.g. EUR)
    """
    logger.info(f"Converting {amount} {from_currency} to {to_currency}")
    return f"{amount} {from_currency} = {amount * 0.85} {to_currency}"


class TravelAgent(Agent):
    def __init__(self, use_tool_proxy: bool = True) -> None:
        toolset_cls = ToolProxyToolset if use_tool_proxy else ToolSearchToolset
        super().__init__(
            instructions="""
You are a comprehensive travel planning assistant with access to multiple specialized
toolsets. Your role is to help users plan trips by providing weather information,
searching and booking flights, finding and reserving hotels, and converting currencies.

Tool Discovery: You have access to a `tool_search` function that lets you discover
available tools.

Voice Interaction Style: This is a voice conversation, not text chat. Keep your responses
short and natural — one or two sentences at a time. Do not list multiple options in a
single response; instead, mention the top choice and ask if the user wants to hear more.
Gather information one piece at a time rather than asking multiple questions at once.

Remember to use tool_search to find the right tools before trying to help the user.
""",
            tools=[
                toolset_cls(
                    id="travel_tools",
                    tools=[
                        WeatherToolset(id="weather"),
                        FlightToolset(id="flights"),
                        HotelToolset(id="hotels"),
                        convert_currency,
                    ],
                    max_results=3,
                )
            ],
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user and let them know you can help with "
            "travel planning: weather, flights, hotels, and currency exchange."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        stt=inference.STT("deepgram/nova-3"),
        vad=silero.VAD.load(),
    )

    # Track token usage to observe prompt caching behavior.
    # With ToolProxyToolset, the tool list is constant, so prompt_cached_tokens
    # should increase after the first turn as the provider caches tool definitions.
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

        # Log cache hit ratio for LLM requests
        if isinstance(ev.metrics, LLMMetrics) and ev.metrics.prompt_tokens > 0:
            metrics.log_metrics(ev.metrics)
            cache_ratio = ev.metrics.prompt_cached_tokens / ev.metrics.prompt_tokens
            logger.info(
                f"Prompt cache: {ev.metrics.prompt_cached_tokens}/{ev.metrics.prompt_tokens} "
                f"tokens cached ({cache_ratio:.0%})"
            )

    await session.start(
        agent=TravelAgent(use_tool_proxy=True),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
