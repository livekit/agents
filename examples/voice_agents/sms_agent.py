import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, override

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    EffectCall,
    JobContext,
    JobExecutorType,
    RunContext,
    TextMessageContext,
    cli,
)
from livekit.agents.beta.workflows import GetEmailTask
from livekit.agents.llm import ToolFlag, function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# uncomment to enable Krisp background voice/noise cancellation
# from livekit.plugins import noise_cancellation

logger = logging.getLogger("basic-agent")

load_dotenv()

PORT = int(os.getenv("PORT", 8081))


class MyAgent(Agent):
    def __init__(self, *, text_mode: bool) -> None:
        super().__init__(
            instructions="Your name is Kelly. You would interact with users via voice."
            "with that in mind keep your responses concise and to the point."
            "do not use emojis, asterisks, markdown, or other special characters in your responses."
            "You are curious and friendly, and have a sense of humor."
            "you will speak english to the user",
        )
        self._text_mode = text_mode

    @override
    def get_init_kwargs(self) -> dict[str, Any]:
        return {
            "text_mode": self._text_mode,
        }

    async def on_enter(self):
        if not self._text_mode:
            logger.debug("greeting the user")
            self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def get_weather(
        self,
        latitude: str,
        longitude: str,
    ):
        """Called when the user asks about the weather. This function will return the weather for
        the given location. When given a location, please estimate the latitude and longitude of the
        location and do not ask the user for them.

        Args:
            latitude: The latitude of the location
            longitude: The longitude of the location
        """

        logger.info(f"getting weather for {latitude}, {longitude}")
        self.session.say("I'm getting the weather for you...")

        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
        weather_data = {}
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # response from the function call is returned to the LLM
                    weather_data = {
                        "temperature": data["current"]["temperature_2m"],
                        "temperature_unit": "Celsius",
                    }
                else:
                    raise Exception(f"Failed to get weather data, status code: {response.status}")

        return weather_data

    @function_tool(flags=ToolFlag.DURABLE)
    async def register_for_weather(self, context: RunContext):
        """Called when the user wants to register for the weather event."""
        logger.info("register_for_weather called")

        get_email_task = GetEmailTask(
            extra_instructions=(
                "You are communicate to the user via text messages, "
                "so there is no need to verify the email address with the user multiple times."
            )
            if self._text_mode
            else "",
            chat_ctx=self.chat_ctx.copy(
                exclude_function_call=True, exclude_instructions=True, exclude_config_update=True
            ),
        )

        email_result = await EffectCall(get_email_task)
        email_address = email_result.email_address

        logger.info(f"User's email address: {email_address}")

        return "You are now registered for the weather event."
        # context.session.say("You are now registered for the weather event.")


server = AgentServer(port=PORT, job_executor_type=JobExecutorType.THREAD)


@server.text_handler(endpoint="weather")
async def text_handler(ctx: TextMessageContext):
    logger.info(f"text message received: {ctx.text}", extra={"session_id": ctx.session_id})

    session = AgentSession(
        llm="openai/gpt-4.1-mini",
        # state_passphrase="my-secret-passphrase",
    )

    start_result = await session.start(
        agent=ctx.session_state if ctx.session_state else MyAgent(text_mode=True),
        capture_run=True,
        wait_run_state=False,
    )
    async for ev in start_result:
        await ctx.send_response(ev)

    logger.info(f"running session with text input: {ctx.text}")
    async for ev in session.run(user_input=ctx.text):
        await ctx.send_response(ev)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        preemptive_generation=True,
    )

    await session.start(agent=MyAgent(text_mode=False), room=ctx.room)


# --- HTTP endpoint examples ---


@server.http_endpoint("/api/user/{user_id}", methods=["GET"])
async def get_user(user_id: str) -> dict:
    """GET with query-parameter parsing.

    Query parameters are mapped to function arguments by name.
    Return a dict and the framework serialises it as JSON.

    Example:
        GET /api/user/abc123
    """
    # replace with a real DB lookup
    return {"user_id": user_id, "name": "Jane Doe", "plan": "pro"}


@server.http_endpoint(
    "/api/stream",
    methods=["GET"],
    headers={"Content-Type": "text/event-stream"},
)
async def stream_events() -> AsyncIterator[str]:
    """Streaming response via an async iterator.

    Yield strings (or bytes) and the framework streams them chunk-by-chunk
    to the client.  Useful for SSE, progress updates, or large payloads.

    Example:
        GET /api/stream
    """

    async def _generate():
        for i in range(5):
            yield f"data: event {i}\n\n"
            await asyncio.sleep(0.5)

    return _generate()


@server.http_endpoint("/api/upload", methods=["POST"])
async def upload_file(request: web.Request) -> web.Response:
    """Raw request / response — full control over the HTTP exchange.

    Accept `web.Request` to read headers, multipart uploads, raw bytes, etc.
    Return a `web.Response` (or `web.StreamResponse`) for custom status codes,
    headers, and body.

    Example:
        POST /api/upload  (multipart form with a "file" field)
    """
    reader = await request.multipart()
    field = await reader.next()

    if field is None or field.name != "file":
        return web.json_response({"error": "missing 'file' field"}, status=400)

    contents = await field.read()  # type: ignore
    filename = field.filename or "upload"

    # replace with real storage (S3, GCS, local disk, etc.)
    logger.info(f"received file '{filename}' ({len(contents)} bytes)")

    return web.json_response(
        {"ok": True, "filename": filename, "size": len(contents)},
        status=201,
    )


if __name__ == "__main__":
    cli.run_app(server)
