"""
Weather Agent - Demonstrates HTTP tool usage with real APIs

This agent can:
- Search for locations by name
- Get current weather conditions
- Provide 7-day weather forecasts

Uses free Open-Meteo API (no API key required)
"""

import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.beta.tools.http import HTTPToolConfig, create_http_tool
from livekit.agents.beta.tools.sms import SMSToolConfig, create_sms_tool
from livekit.plugins import silero

logger = logging.getLogger("weather-agent")

load_dotenv()


def _format_location_response(resp) -> str:
    """Parse geocoding response and return formatted location info."""
    import json

    data = json.loads(resp.body)

    if not data.get("results"):
        return "Sorry, I couldn't find that location."

    result = data["results"][0]
    return json.dumps(
        {
            "name": result.get("name"),
            "latitude": result.get("latitude"),
            "longitude": result.get("longitude"),
            "country": result.get("country"),
            "timezone": result.get("timezone"),
        }
    )


def _format_current_weather(resp) -> str:
    """Parse current weather response into readable format."""
    import json

    data = json.loads(resp.body)
    current = data.get("current", {})

    # WMO Weather interpretation codes
    weather_codes = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    weather_desc = weather_codes.get(current.get("weather_code", 0), "Unknown")
    temp = current.get("temperature_2m")
    feels_like = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind = current.get("wind_speed_10m")

    return (
        f"Current weather: {weather_desc}. "
        f"Temperature is {temp}°C (feels like {feels_like}°C). "
        f"Humidity at {humidity}%, wind speed {wind} km/h."
    )


def _format_forecast(resp) -> str:
    """Parse forecast response into readable format."""
    import json

    data = json.loads(resp.body)
    daily = data.get("daily", {})

    # just show first 3 days to keep it concise
    times = daily.get("time", [])[:3]
    max_temps = daily.get("temperature_2m_max", [])[:3]
    min_temps = daily.get("temperature_2m_min", [])[:3]

    forecast_text = "Here's the forecast: "
    for i, date in enumerate(times):
        forecast_text += f"{date}: High {max_temps[i]}°C, Low {min_temps[i]}°C. "

    return forecast_text


# Create HTTP tools for weather APIs
search_location_tool = create_http_tool(
    HTTPToolConfig(
        name="search_location",
        description="Search for a location by name to get its coordinates. Always use this first when the user mentions a city name.",
        url="https://geocoding-api.open-meteo.com/v1/search",
        method="GET",
        timeout_ms=5000,
        parameters={
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "City name to search for (e.g., 'New York', 'London', 'Tokyo')",
                },
                "count": {
                    "type": "number",
                    "description": "Number of results to return. Default is 1 (best match).",
                },
                "language": {
                    "type": "string",
                    "description": "Language for results (en, ru, etc). Default is 'en'.",
                },
            },
        },
        output_normalizer=_format_location_response,
    )
)


get_current_weather_tool = create_http_tool(
    HTTPToolConfig(
        name="get_current_weather",
        description="Get the current weather for a specific location. Returns temperature, humidity, apparent temperature, weather condition, and wind speed.",
        url="https://api.open-meteo.com/v1/forecast",
        method="GET",
        timeout_ms=10000,
        parameters={
            "type": "object",
            "required": ["latitude", "longitude"],
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude coordinate (e.g., 40.7128 for New York, 51.5074 for London)",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude coordinate (e.g., -74.0060 for New York, -0.1278 for London)",
                },
                "current": {
                    "type": "string",
                    "description": "Current weather variables to fetch",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone (use 'auto' to detect automatically)",
                },
                "temperature_unit": {
                    "type": "string",
                    "description": "Temperature unit: 'celsius' or 'fahrenheit'",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
        },
        output_normalizer=_format_current_weather,
    )
)


get_weather_forecast_tool = create_http_tool(
    HTTPToolConfig(
        name="get_weather_forecast",
        description="Get a 7-day weather forecast for a location. Returns daily high/low temperatures, weather conditions, precipitation, and wind speed.",
        url="https://api.open-meteo.com/v1/forecast",
        method="GET",
        timeout_ms=15000,
        parameters={
            "type": "object",
            "required": ["latitude", "longitude"],
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude coordinate (e.g., 40.7128 for New York)",
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude coordinate (e.g., -74.0060 for New York)",
                },
                "daily": {
                    "type": "string",
                    "description": "Daily weather variables to fetch",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone (use 'auto' to detect automatically)",
                },
                "temperature_unit": {
                    "type": "string",
                    "description": "Temperature unit: 'celsius' or 'fahrenheit'",
                    "enum": ["celsius", "fahrenheit"],
                },
                "forecast_days": {
                    "type": "number",
                    "description": "Number of forecast days (1-16). Default is 7.",
                },
            },
        },
        execution_message="One moment, fetching the weather forecast for you.",
        output_normalizer=_format_forecast,
    )
)


# Create SMS tool (automatically sends to the caller)
# Only initialize if SMS provider credentials are available
# It will skip sending the SMS if no credentials are found
try:
    send_weather_sms_tool = create_sms_tool(
        SMSToolConfig(
            name="send_weather_sms",
            description="Send weather information via SMS to the caller. Use this when the user asks to receive weather info by text message.",
            to_number="+15555555555",  # PUT YOUR PHONE NUMBER HERE
        )
    )
except ValueError:
    logger.warning("SMS tool not available: no SMS provider credentials found")
    send_weather_sms_tool = None


class WeatherAgent(Agent):
    def __init__(self) -> None:
        tools = [
            search_location_tool,
            get_current_weather_tool,
            get_weather_forecast_tool,
        ]

        instructions = (
            "You are a helpful weather assistant. When a user asks about weather:\n"
            "1. First use search_location to get coordinates for the city\n"
            "2. Then use get_current_weather or get_weather_forecast with those coordinates\n"
            "3. Be friendly and conversational\n"
            "4. Always mention the city name in your response\n"
        )

        if send_weather_sms_tool:
            tools.append(send_weather_sms_tool)
            instructions += "5. If the user asks to receive weather info by SMS/text, use send_weather_sms with a concise weather summary\n"

        instructions += (
            "Note: Open-Meteo API requires specific parameters - for current weather, pass "
            "current='temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m' "
            "and timezone='auto'. For forecast, pass "
            "daily='temperature_2m_max,temperature_2m_min,weather_code,precipitation_sum,wind_speed_10m_max' "
            "and timezone='auto'."
        )

        super().__init__(instructions=instructions)
        self._tools = tools

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="say hello to the user in English and ask them what city they would like to know about"
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-3:a167e0f3-df7e-4d52-a9c3-f949145efdab",
        vad=silero.VAD.load(),
    )

    await session.start(agent=WeatherAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
