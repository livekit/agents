import asyncio
import logging

from livekit import rtc
from livekit.agents import JobContext, JobRequest, VoiceAssistant, WorkerOptions, cli
from livekit.agents.llm import (
    ChatMessage,
    ChatRole,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero


async def entrypoint(ctx: JobContext):
    vad = silero.VAD()
    stt = deepgram.STT()
    llm = openai.LLM()
    tts = elevenlabs.TTS()
    assistant = VoiceAssistant(vad, stt, llm, tts, debug=True, plotting=False)

    # registering the callback in case the participant is not already here
    @ctx.room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant):
        assistant.start(ctx.room, participant)

    # start with the first participant that's already in the room
    for participant in ctx.room.participants.values():
        assistant.start(ctx.room, participant)
        break

    # set system prompt
    assistant.chat_context.messages.append(
        ChatMessage(
            role=ChatRole.SYSTEM,
            text="You are a voice assistant created by Live Kit. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation.",
        )
    )

    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
