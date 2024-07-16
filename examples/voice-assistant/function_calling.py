import asyncio
import copy
import enum
import logging
from typing import Annotated

from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice_assistant import AssistantCallContext, VoiceAssistant
from livekit.plugins import deepgram, elevenlabs, openai, silero

ASSISANT_PROMPT = llm.ChatContext(
    messages=[
        llm.ChatMessage(
            role=llm.ChatRole.SYSTEM,
            text="You are a voice assistant created by LiveKit. Your interface with users will be voice. \
            You should use short and concise responses, and avoiding usage of unpronouncable punctuation.",
        )
    ]
)


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

        # Default state of the lights in each room
        # (for demonstration purposes)
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
    @llm.ai_callable(
        desc="Turn on/off the lights in a room",
        # When enabling auto_retry, the AI function will be automatically retried if the arguments were not
        # provided/invalid. (This is done by prompting the LLM the missing arguments/invalid arguments).
        auto_retry=True,
    )
    async def toggle_light(
        self,
        room: Annotated[Room, llm.TypeInfo(desc="The specific room")],
        status: bool,
    ):
        logging.info("toggle_light - room: %s status: %s", room, status)
        self._light_status[room] = status

        # Store the new state of the room inside the assistant call context.
        # An AI function can be executed multiple times concurrently, therefore we
        # answer to the user inside "function_calls_finished" event.
        # (OAI supports parallel execution of function calls)
        call_ctx = AssistantCallContext.get_current()
        toggled_rooms = call_ctx.get_metadata("toggled_rooms", [])
        toggled_rooms.append((room, status))
        call_ctx.store_metadata("toggled_rooms", toggled_rooms)

    # Use the python documentation as the AI function description
    @llm.ai_callable(desc=llm.USE_DOCSTRING)
    def stop_speaking(self):
        """
        Call this function when the user want the assistant to stop speaking.
        The user can interrupt the assistant at any time.
        """
        # This function does nothing, by calling this AI  function,
        # the LLM will not generate any text output.
        pass


async def entrypoint(ctx: JobContext):
    fnc_ctx = AssistantFnc()  # create our fnc ctx instance

    # AI functions can also be added at runtime if needed
    # example data (e.g this could be loaded from a database)
    contacts = [
        "Alice",
        "Bob",
        "Theo",
    ]

    @fnc_ctx.ai_callable(desc="Sends a text message to a specified contact.")
    async def send_message(
        contact: Annotated[
            str,
            llm.TypeInfo(
                desc="The name or identifier of the contact to whom the message will be sent.",
                choices=contacts,
            ),
        ],
        message: str,
    ):
        logging.info("sending a message to %s: %s", contact, message)
        call_ctx = AssistantCallContext.get_current()
        messages_sent = call_ctx.get_metadata("messages_sent", [])
        messages_sent.append((contact, message))
        call_ctx.store_metadata("messages_sent", messages_sent)

    # create our VoiceAssistant instance

    async def _inject_current_ctx_llm(
        assistant: VoiceAssistant, chat_ctx: llm.ChatContext
    ):
        # inject the current state into the context of the LLM
        state_msg = llm.ChatMessage(
            role=llm.ChatRole.SYSTEM,
            text=(
                "The current state of the lights is:\n"
                + "\n".join(
                    f"{room.value}: {'on' if status else 'off'}"
                    for room, status in fnc_ctx.light_status.items()
                )
            ),
        )
        chat_ctx = copy.deepcopy(chat_ctx)
        chat_ctx.messages.append(state_msg)
        return await assistant.llm.chat(chat_ctx=chat_ctx, fnc_ctx=assistant.fnc_ctx)

    gpt4o = openai.LLM(model="gpt-4o")
    assistant = VoiceAssistant(
        vad=silero.VAD(),
        stt=deepgram.STT(),
        llm=gpt4o,
        tts=elevenlabs.TTS(),
        fnc_ctx=fnc_ctx,
        chat_ctx=ASSISANT_PROMPT,
        will_create_llm_stream=_inject_current_ctx_llm,
    )

    @assistant.on("agent_speech_interrupted")
    def agent_speech_interrupted(
        chat_ctx: llm.ChatContext, assistant_msg: llm.ChatMessage
    ):
        # the user interrupted the assistant
        # reflect that in the LLM context by editing the assistant_message
        assistant_msg.text += "... (user interrupted you)"

    @assistant.on("function_calls_finished")
    def function_calls_finished(call_ctx: AssistantCallContext):
        asyncio.create_task(_answer_actions(call_ctx))

    # Start the assisant, this will automatically publish a microphone track
    # as well as listening to the first participant it founds inside the current room.
    # If you need to specify a particular participant, you can use the `participant` parameter.
    assistant.start(ctx.room)

    await asyncio.sleep(3)
    await assistant.say("Hey, how can I help you today?")


async def _answer_actions(
    call_ctx: AssistantCallContext,
):
    toggled_rooms = call_ctx.get_metadata("toggled_rooms", [])
    messages_sent = call_ctx.get_metadata("messages_sent", [])
    if not toggled_rooms and not messages_sent:
        return

    prompt = "Make a summary of the following actions you did:"
    if len(toggled_rooms) > 0:
        enabled_rooms = [room.value for room, status in toggled_rooms if status is True]
        disabled_rooms = [
            room.value for room, status in toggled_rooms if status is False
        ]

        enabled_rooms_str = ", ".join(enabled_rooms)
        prompt += (
            f"\n - You enabled the lights in the following rooms: {enabled_rooms_str}"
        )

        disabled_rooms_str = ", ".join(disabled_rooms)
        prompt += (
            f"\n - You disabled the lights in the following rooms {disabled_rooms_str}"
        )

    if len(messages_sent) > 0:
        prompt += "\n - You sent some messages to the following contacts:"
        for contact, message in messages_sent:
            prompt += f"\n - {contact}"

    chat_ctx = llm.ChatContext(
        messages=[llm.ChatMessage(role=llm.ChatRole.SYSTEM, text=prompt)]
    )

    stream = await call_ctx.assistant.llm.chat(chat_ctx=chat_ctx)
    await call_ctx.assistant.say(stream)


async def request_fnc(req: JobRequest) -> None:
    logging.info("received request %s", req)
    await req.accept(entrypoint)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(request_fnc))
