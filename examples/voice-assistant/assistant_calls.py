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
)
from livekit.agents.voice_assistant import VoiceAssistant, AssistantContext
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
        ctx = AssistantContext.get_current()
        key = "enabled_rooms" if status else "disabled_rooms"
        li = ctx.get_metadata(key, [])
        ctx.store_metadata(key, li)

    @llm.ai_callable(desc="User want the assistant to stop/pause speaking")
    def stop_speaking(self):
        pass  # do nothing


async def entrypoint(ctx: JobContext):
    gpt = openai.LLM(model="gpt-4-turbo")

    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=elevenlabs.TTS(),
        fnc_ctx=AssistantFnc(),
        debug=True,
    )

    @assistant.on("agent_speech_interrupted")
    def _agent_speech_interrupted(chat_ctx: llm.ChatContext, msg: llm.ChatMessage):
        msg.text += "... (user interrupted you)"

    @assistant.on("function_calls_done")
    def _function_calls_done(ctx: AssistantContext):
        enabled_rooms = ctx.get_metadata("enabled_rooms", [])
        disabled_rooms = ctx.get_metadata("disabled_rooms", [])

        async def _handle_answer():
            prompt = "You disabled the lights in the following rooms: " + ", ".join(
                disabled_rooms
            )
            prompt += "\nYou enabled the lights in the following rooms: " + ", ".join(
                enabled_rooms
            )

            messages = assistant.chat_context.messages.copy()
            messages.append(llm.ChatMessage(role=llm.ChatRole.SYSTEM, text=prompt))
            chat_ctx = llm.ChatContext(messages=messages)

            gpt_stream = await gpt.chat(chat_ctx)
            await assistant.say(gpt_stream)

        asyncio.ensure_future(_handle_answer())

    # start the assistant on the first participant found

    @ctx.room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant):
        assistant.start(ctx.room, participant)

    for participant in ctx.room.participants.values():
        assistant.start(ctx.room, participant)
        break

    await asyncio.sleep(1)
    await assistant.say("Hey, how can I help you today?")


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
