from livekit.agents.llm import function_tool


@function_tool
async def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to check weather for.
    """
    # In a real implementation, this would call a weather API
    return f"The weather in {city} is 72°F and sunny with light winds."
