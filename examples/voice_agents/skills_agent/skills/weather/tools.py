from livekit.agents.llm import function_tool


@function_tool
async def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: The city name.
    """
    return f"Sunny, 72F in {city}"


@function_tool
async def get_forecast(city: str, days: int) -> str:
    """Get weather forecast for upcoming days.

    Args:
        city: The city name.
        days: Number of days to forecast.
    """
    return f"{days}-day forecast for {city}: mostly sunny"
