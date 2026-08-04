import asyncio
import random

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("Demo 🚀")


@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"The weather in {location} is a perfect sunny 70°F today. Enjoy your day!"


@mcp.tool()
async def book_flight(origin: str, destination: str, date: str, ctx: Context) -> str:
    """Book a flight. Takes a couple of minutes — emits progress notifications
    while searching airlines and confirming the booking."""
    await ctx.report_progress(
        0.0,
        1.0,
        f"Searching flights from {origin} to {destination} on {date}. "
        "This will take a couple of minutes.",
    )

    # Phase 1: searching airlines
    await asyncio.sleep(30)

    airlines = random.sample(["United", "Delta", "American", "JetBlue", "Southwest", "Alaska"], k=3)
    prices = {a: random.randint(180, 650) for a in airlines}
    cheapest = min(prices, key=lambda a: prices[a])

    await ctx.report_progress(
        0.5,
        1.0,
        f"Found {len(airlines)} options. Best price: ${prices[cheapest]} on {cheapest}. "
        "Confirming the booking now.",
    )

    # Phase 2: confirming booking
    await asyncio.sleep(40)

    confirmation = f"FL-{random.randint(100000, 999999)}"
    return (
        f"Flight booked! {cheapest} from {origin} to {destination} on {date}. "
        f"Price: ${prices[cheapest]}. Confirmation: {confirmation}."
    )


if __name__ == "__main__":
    mcp.run(transport="sse")
