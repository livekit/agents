import asyncio
import logging

from livekit import rtc
from livekit.agents import JobContext, JobRequest, VoiceAssistant, WorkerOptions, cli
from livekit.plugins import deepgram, elevenlabs, openai, silero


async def entrypoint(ctx: JobContext):
    vad = silero.VAD()
    stt = deepgram.STT()
    llm = openai.LLM()
    tts = elevenlabs.TTS()
    assistant = VoiceAssistant(vad, stt, llm, tts, debug=True, plotting=True)

    @ctx.room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant):
        assistant.start(ctx.room, participant)

    for participant in ctx.room.participants.values():
        assistant.start(ctx.room, participant)
        break

    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?", allow_interruptions=True)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
