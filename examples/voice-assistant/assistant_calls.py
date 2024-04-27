import asyncio
import logging
from enum import Enum
from typing import Annotated

from livekit import rtc
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    llm,
    voice_assistant,
)
from livekit.plugins import deepgram, elevenlabs, openai, silero


class Room(Enum):
    BEDROOM = "bedroom"
    LIVING_ROOM = "living room"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"


class AssistantFnc(llm.FunctionContext):
    @llm.ai_callable(desc="Turn on/off the lights in a room")
    async def toggle_light(
        self,
        room: Annotated[Room, llm.TypeInfo(desc="The specific room")],
        status: bool,
    ):
        print(f"Turning the lights in {room} {'on' if status else 'off'}")
        pass


async def entrypoint(ctx: JobContext):
    fnc_ctx = AssistantFnc()

    vad = silero.VAD()
    stt = deepgram.STT()
    llm = openai.LLM()
    tts = elevenlabs.TTS()
    assistant = voice_assistant.VoiceAssistant(
        vad, stt, llm, tts, fnc_ctx=fnc_ctx, debug=True, plotting=True
    )

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
