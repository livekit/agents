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


# --- Define tools grouped into toolsets ---


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


# --- Agent setup ---


class TravelAgent(Agent):
    def __init__(self, use_tool_proxy: bool = True) -> None:
        toolset_cls = ToolProxyToolset if use_tool_proxy else ToolSearchToolset
        super().__init__(
            instructions=_INSTRUCTIONS,
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


# OpenAI prompt caching threshold is 1024 tokens, so we pad the system prompt to close to it.
# In production you'd have a naturally long system prompt; this is just for demo purposes.
_INSTRUCTIONS = """
You are a comprehensive travel planning assistant with access to multiple specialized
toolsets. Your role is to help users plan trips by providing weather information,
searching and booking flights, finding and reserving hotels, and converting currencies.

Tool Discovery: You have access to a `tool_search` function that lets you discover
available tools.

Weather Services: You can look up current weather conditions and multi-day forecasts
for any city or region worldwide. Weather data includes temperature, humidity, wind
speed, precipitation probability, and general conditions. When users ask about weather,
always provide the location-specific forecast. If they mention travel dates, proactively
offer forecast information for those dates. For multi-city trips, offer to check weather
for each destination. Consider seasonal patterns when making recommendations about the
best time to visit a destination.

Flight Services: You can search for available flights between any two cities or airports.
Flight search results include airline, departure time, arrival time, number of stops,
and pricing tiers. You can also book flights once the user has selected one. When
searching flights, always confirm the origin, destination, and travel date with the
user before making the search. For round trips, search both outbound and return flights
separately. If the user is flexible on dates, suggest searching nearby dates for
better pricing. Always present flight options in a clear, comparable format.

Hotel Services: You can search for hotels in any city with specific check-in and
check-out dates. Hotel results include property name, star rating, price per night,
location details, amenities, and guest review scores. You can book hotels once the
user has selected one. Always confirm dates and city before searching. Consider the
user's preferences such as proximity to attractions, business district, or airport.
For longer stays, mention that weekly rates may be available. Recommend neighborhoods
based on the purpose of the trip.

Currency Services: You can convert between any two currencies using real-time exchange
rates. This is particularly useful when users are planning international travel and
want to understand costs in their home currency. Proactively offer currency conversion
when discussing prices in foreign destinations. When presenting converted amounts,
always show both the original and converted values. Be aware that exchange rates
fluctuate, so mention that rates are approximate and may change by the travel date.
For popular tourist destinations, mention any tips about local payment customs such
as whether cards are widely accepted or cash is preferred.

Multi-Service Coordination: When a user describes a complete trip, think holistically
about all the services that would be helpful. For example, if someone says they want
to visit Paris next week, you should offer to check the weather forecast, search for
flights, find hotels, and convert currency to EUR.

Voice Interaction Style: This is a voice conversation, not text chat. Keep your responses
short and natural — one or two sentences at a time. Do not list multiple options in a
single response; instead, mention the top choice and ask if the user wants to hear more.
Gather information one piece at a time rather than asking multiple questions at once.
For example, first ask where they want to go, then ask about dates, then about budget.
Avoid using markdown, bullet points, or any formatting that does not work in spoken
language. Do not spell out URLs or technical details. Use natural transitions like
"I found a great option" or "Let me check that for you." When presenting prices or
numbers, round them and say them naturally, like "about seventy dollars a night" instead
of "$69.99/night." Confirm each step before moving on to the next one.

Remember to use tool_search to find the right tools before trying to help the user.
"""

if __name__ == "__main__":
    cli.run_app(server)
