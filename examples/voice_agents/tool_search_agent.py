"""Example: ToolSearchToolset for dynamic tool loading.

Instead of loading all tools into the LLM context upfront, ToolSearchToolset exposes
a single `tool_search` function. When the LLM needs a capability, it searches for
matching tools which are then dynamically loaded into the context.

This is useful when you have many tools (10+) and want to reduce context size
and improve tool selection accuracy.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, llm
from livekit.agents.beta.toolsets import ToolSearchToolset
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
        return f"Sunny, 72°F in {location}"

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
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a travel assistant. You can help with weather, flights, "
                "hotels, and currency conversion. Use tool_search to find the right "
                "tools before trying to help the user."
            ),
            tools=[
                ToolSearchToolset(
                    id="travel_tools",
                    tools=[
                        WeatherToolset(id="weather"),
                        FlightToolset(id="flights"),
                        HotelToolset(id="hotels"),
                        convert_currency,
                    ],
                    max_results=3,
                ),
            ],
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user and let them know you can help with "
            "travel planning — weather, flights, hotels, and currency exchange."
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

    await session.start(agent=TravelAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
