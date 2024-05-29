import asyncio
import logging

from livekit.agents import JobContext, JobRequest, WorkerOptions, cli
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ChatRole,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, elevenlabs, openai, silero


async def entrypoint(ctx: JobContext):
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                text="You are a voice assistant created by LiveKit. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation.",
            )
        ]
    )

    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=elevenlabs.TTS(),
        chat_ctx=initial_ctx,
    )
    assistant.start(ctx.room)

    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
