import asyncio
import logging
from enum import Enum
from typing import Annotated

from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice_assistant import AssistantContext, VoiceAssistant
from livekit.plugins import deepgram, elevenlabs, openai, silero


class Room(Enum):
    BEDROOM = "bedroom"
    LIVING_ROOM = "living room"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"


class AssistantFnc(llm.FunctionContext):
    @llm.ai_callable(desc="Turn on/off the lights in a room")
    async def toggle_light(
        self,
        room: Annotated[Room, llm.TypeInfo(desc="The specific room")],
        status: bool,
    ):
        logging.info("toggle_light %s %s", room, status)
        ctx = AssistantContext.get_current()
        key = "enabled_rooms" if status else "disabled_rooms"
        li = ctx.get_metadata(key, [])
        li.append(room)
        ctx.store_metadata(key, li)

    @llm.ai_callable(desc="User want the assistant to stop/pause speaking")
    def stop_speaking(self):
        pass  # do nothing


async def entrypoint(ctx: JobContext):
    gpt = openai.LLM(model="gpt-4-turbo")

    initial_ctx = llm.ChatContext(
        messages=[
            llm.ChatMessage(
                role=llm.ChatRole.SYSTEM,
                text="You are a voice assistant created by LiveKit. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation.",
            )
        ]
    )

    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=elevenlabs.TTS(),
        fnc_ctx=AssistantFnc(),
        chat_ctx=initial_ctx,
    )

    async def _answer_light_toggling(enabled_rooms, disabled_rooms):
        prompt = "Make a summary of the following actions you did:"
        if enabled_rooms:
            enabled_rooms_str = ", ".join(enabled_rooms)
            prompt += f"\n - You enabled the lights in the following rooms: {enabled_rooms_str}"

        if disabled_rooms:
            disabled_rooms_str = ", ".join(disabled_rooms)
            prompt += f"\n - You disabled the lights in the following rooms {disabled_rooms_str}"

        chat_ctx = llm.ChatContext(
            messages=[llm.ChatMessage(role=llm.ChatRole.SYSTEM, text=prompt)]
        )

        stream = await gpt.chat(chat_ctx)
        await assistant.say(stream)

    @assistant.on("agent_speech_interrupted")
    def _agent_speech_interrupted(chat_ctx: llm.ChatContext, msg: llm.ChatMessage):
        msg.text += "... (user interrupted you)"

    @assistant.on("function_calls_finished")
    def _function_calls_done(ctx: AssistantContext):
        logging.info("function_calls_done %s", ctx)
        enabled_rooms = ctx.get_metadata("enabled_rooms", [])
        disabled_rooms = ctx.get_metadata("disabled_rooms", [])

        if enabled_rooms or disabled_rooms:
            # if there was a change in the lights, summarize it and let the user know
            asyncio.ensure_future(_answer_light_toggling(enabled_rooms, disabled_rooms))

    assistant.start(ctx.room)
    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?")


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
