import json
import logging
from dataclasses import dataclass
from typing import Annotated

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.plugins import deepgram, openai, silero

load_dotenv()

logger = logging.getLogger("multi-stage-agent")
logger.setLevel(logging.INFO)


@dataclass
class AgentSpec:
    instructions: str
    fnc_ctx: llm.FunctionContext

    @classmethod
    def create(cls, instructions: str, fncs: dict[str, llm.FunctionInfo]):
        spec = cls(instructions=instructions, fnc_ctx=llm.FunctionContext())
        spec.fnc_ctx._fncs.update(fncs)
        return spec


class RestaurantBot(llm.FunctionContext):
    def __init__(self):
        super().__init__()

        self._specs = {
            "Greeter": AgentSpec.create(
                instructions=(
                    "You are a professional restaurant receptionist handling incoming calls. "
                    "Warmly greet the caller and ask if they would like to place an order. "
                    "Available menu items: Pizza, Salad, Ice Cream, Coffee. "
                    "Guide the conversation as follows:\n"
                    "- If they want to place an order, transfer them to order taking\n"
                    "- If they have completed their order, transfer them to customer details\n"
                    "- For any other inquiries, assist them directly\n"
                    "Maintain a friendly and professional tone throughout the conversation."
                    "Use the functions to transfer the call to the next step."
                ),
                fncs={
                    "transfer_to_ordering": self._fncs["transfer_to_ordering"],
                    "transfer_to_info_collection": self._fncs[
                        "transfer_to_info_collection"
                    ],
                },
            ),
            "OrderTaking": AgentSpec.create(
                instructions=(
                    "You are a professional order taker at a restaurant. "
                    "Guide the customer through their order with these steps:\n"
                    "1. Take their order selections one at a time from our menu: Pizza, Salad, Ice Cream, Coffee\n"
                    "2. Clarify any special requests or modifications\n"
                    "3. Repeat back the complete order to confirm accuracy\n"
                    "4. Once confirmed, transfer them back to the greeter\n"
                    "Be attentive and ensure order accuracy before proceeding."
                ),
                fncs={
                    "take_order": self._fncs["take_order"],
                    "transfer_to_greeter": self._fncs["transfer_to_greeter"],
                },
            ),
            "CustomerDetails": AgentSpec.create(
                instructions=(
                    "You are collecting essential customer information for their order. "
                    "Follow these steps carefully:\n"
                    "1. Ask for the customer's name and confirm the spelling\n"
                    "2. Request their phone number and verify it's correct\n"
                    "3. Repeat both pieces of information back to ensure accuracy\n"
                    "4. Once confirmed, transfer back to the greeter\n"
                    "Handle personal information professionally and courteously."
                ),
                fncs={
                    "collect_name": self._fncs["collect_name"],
                    "collect_phone": self._fncs["collect_phone"],
                    "transfer_to_greeter": self._fncs["transfer_to_greeter"],
                },
            ),
        }

        self._cur_spec = self._specs["Greeter"]

    def _transfer_to_spec(self, spec_name: str, agent: VoicePipelineAgent) -> None:
        self._cur_spec = self._specs[spec_name]
        # TODO: update chat ctx for each spec
        # agent._chat_ctx = self.get_chat_ctx(agent._chat_ctx)
        logger.info(f"Transferring to {spec_name}")

    def get_chat_ctx(self, chat_ctx: llm.ChatContext | None = None) -> llm.ChatContext:
        """Get the chat context for the current spec"""
        new_chat_ctx = llm.ChatContext().append(
            text=self._cur_spec.instructions,
            role="system",
        )
        if chat_ctx:
            messages = chat_ctx.messages
            if messages and messages[0].role == "system":
                messages = messages[1:]

            # # Greeter has all the chat history, others have the last 6 messages
            # if self._cur_spec != "Greeter":
            #     messages = messages[-6:]
            new_chat_ctx.messages.extend(messages)

        return new_chat_ctx

    def before_llm_callback(
        self, agent: VoicePipelineAgent, chat_ctx: llm.ChatContext
    ) -> llm.LLMStream:
        return agent.llm.chat(
            chat_ctx=self.get_chat_ctx(chat_ctx),
            fnc_ctx=self._cur_spec.fnc_ctx,
            parallel_tool_calls=False,
        )

    @llm.ai_callable()
    async def take_order(
        self,
        item: Annotated[str, llm.TypeInfo(description="The item added to the order")],
    ):
        """Called when the user orders a new item from our menu."""
        logger.info(f"Taking order for {item}")
        return f"Received order for {item}"

    @llm.ai_callable()
    async def collect_name(
        self, name: Annotated[str, llm.TypeInfo(description="The customer's name")]
    ):
        """Called when the user provides their name."""
        logger.info(f"Collecting name: {name}")
        return f"Please confirm with the customer that their name is {name}."

    @llm.ai_callable()
    async def collect_phone(
        self,
        phone: Annotated[str, llm.TypeInfo(description="The customer's phone number")],
    ):
        """Called when the user provides their phone number."""
        logger.info(f"Collecting phone: {phone}")
        return f"Please confirm with the customer that their phone number is {phone}."

    @llm.ai_callable()
    async def transfer_to_ordering(self):
        """Called to transfer the call to order taking."""
        call_ctx = AgentCallContext.get_current()
        self._transfer_to_spec("OrderTaking", call_ctx.agent)
        return "Transferred to order taking."

    @llm.ai_callable()
    async def transfer_to_info_collection(self):
        """Called to transfer the call to collect the customer's details."""
        call_ctx = AgentCallContext.get_current()
        self._transfer_to_spec("CustomerDetails", call_ctx.agent)
        return "Transferred to collecting customer details."

    @llm.ai_callable()
    async def transfer_to_greeter(self):
        """Called to transfer the call back to the greeter."""
        call_ctx = AgentCallContext.get_current()
        self._transfer_to_spec("Greeter", call_ctx.agent)
        return "Back to the greeter."


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    fnc_ctx = RestaurantBot()
    initial_chat_ctx = fnc_ctx.get_chat_ctx()

    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
        before_llm_cb=fnc_ctx.before_llm_callback,
        # preemptive_synthesis=True,
    )

    @ctx.room.on("data_received")
    def on_data_received(packet: rtc.DataPacket):
        if packet.topic == "lk-chat-topic":
            data = json.loads(packet.data.decode("utf-8"))
            logger.info(f"Text input received: {data}")

            agent._human_input.emit(
                "final_transcript",
                SpeechEvent(
                    type=SpeechEventType.END_OF_SPEECH,
                    alternatives=[SpeechData(language="en", text=data["message"])],
                ),
            )

    # Start the assistant. This will automatically publish a microphone track and listen to the participant.
    agent.start(ctx.room, participant)
    await agent.say(
        "Welcome to our restaurant! How may I assist you with your order today?"
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )
