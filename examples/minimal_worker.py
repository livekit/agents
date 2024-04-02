import logging
import asyncio
from livekit.agents import (
    JobRequest,
    JobContext,
    Worker,
    cli,
    VoiceAssistant,
    WorkerOptions,
)

from livekit.plugins import silero
from livekit.plugins import deepgram
from livekit.plugins import elevenlabs
from livekit.plugins import openai


async def entrypoint(job: JobContext):
    logging.info("sdtarting voice assistant...")

    # blablabla


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc=request_fnc))
