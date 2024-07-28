import asyncio
import enum
import logging
from typing import Annotated

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

logger = logging.getLogger("function-calling-demo")
logger.setLevel(logging.INFO)


class Room(enum.Enum):
    # ai_callable can understand enum types as a set of choices
    # this is equivalent to:
    #     `Annotated[Room, llm.TypeInfo(choices=["bedroom", "living room", "kitchen", "bathroom", "office"])]`
    BEDROOM = "bedroom"
    LIVING_ROOM = "living room"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    OFFICE = "office"


class AssistantFnc(llm.FunctionContext):
    """
    The class defines a set of AI functions that the assistant can execute.
    """

    def __init__(self) -> None:
        super().__init__()

        # default state of the lights in each room
        self._light_status = {
            Room.BEDROOM: False,
            Room.LIVING_ROOM: True,
            Room.KITCHEN: True,
            Room.BATHROOM: False,
            Room.OFFICE: False,
        }

    @property
    def light_status(self):
        return self._light_status

    # Simple demonstration of an AI function that can be called by the user with some arguments.
    @llm.ai_callable(description="Turn on/off the lights in a room")
    async def toggle_light(
        self,
        room: Annotated[Room, llm.TypeInfo(description="The specific room")],
        status: bool,
    ):
        logger.info("toggle_light - room: %s status: %s", room, status)
        self._light_status[room] = status
        return f"Turned the lights in the {room} {'on' if status else 'off'}"


async def entrypoint(ctx: JobContext):
    fnc_ctx = AssistantFnc()  # create our fnc ctx instance

    async def _will_synthesize_assistant_reply(
        assistant: VoiceAssistant, chat_ctx: llm.ChatContext
    ):
        # Inject the current state of the lights into the context of the LLM
        chat_ctx = chat_ctx.copy()
        chat_ctx.messages.append(
            llm.ChatMessage(
                content=(
                    "Current state of the lights:\n"
                    + "\n".join(
                        f"- {room}: {'on' if status else 'off'}"
                        for room, status in fnc_ctx.light_status.items()
                    )
                ),
                role="system",
            )
        )
        return assistant.llm.chat(chat_ctx=chat_ctx, fnc_ctx=assistant.fnc_ctx)

    initial_chat_ctx = llm.ChatContext()
    initial_chat_ctx.messages.append(
        llm.ChatMessage(
            content=(
                "You are a home assistant created by LiveKit. Your interface with users will be voice. "
                "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
            ),
            role="system",
        )
    )

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
        will_synthesize_assistant_reply=_will_synthesize_assistant_reply,
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Start the assistant. This will automatically publish a microphone track and listen to the first participant
    # it finds in the current room. If you need to specify a particular participant, use the participant parameter.
    assistant.start(ctx.room)

    await asyncio.sleep(2)
    await assistant.say("Hey, how can I help you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
