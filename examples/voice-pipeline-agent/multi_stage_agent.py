import json
import logging
from dataclasses import dataclass
from typing import Annotated, Callable

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
    chat_ctx: llm.ChatContext
    fnc_ctx: llm.FunctionContext

    @classmethod
    def create(cls, instructions: str, fncs: list[Callable]):
        chat_ctx = llm.ChatContext().append(text=instructions, role="system")
        fnc_ctx = llm.FunctionContext()
        for fnc in fncs:
            fnc_ctx._register_ai_function(fnc)
        return cls(chat_ctx=chat_ctx, fnc_ctx=fnc_ctx)


class RestaurantBot:
    def __init__(self, menu: str = "Pizza, Salad, Ice Cream, Coffee"):
        self._menu = menu
        self._specs = {
            "Greeter": AgentSpec.create(
                instructions=(
                    "You are a professional restaurant receptionist handling incoming calls. "
                    "Warmly greet the caller and ask if they would like to place an order. "
                    f"Available menu items: {self._menu}. "
                    "Guide the conversation as follows:\n"
                    "- If they want to place an order, transfer them to order taking\n"
                    "- If they have completed their order, transfer them to customer details\n"
                    "- For any other inquiries, assist them directly\n"
                    "Maintain a friendly and professional tone throughout the conversation."
                    "Use the functions to transfer the call to the next step."
                ),
                fncs=[
                    self.transfer_to_ordering,
                    self.transfer_to_info_collection,
                ],
            ),
            "OrderTaking": AgentSpec.create(
                instructions=(
                    "You are a professional order taker at a restaurant. "
                    "Guide the customer through their order with these steps:\n"
                    f"1. Take their order selections one at a time from our menu: {self._menu}\n"
                    "2. Clarify any special requests or modifications\n"
                    "3. Repeat back the complete order to confirm accuracy\n"
                    "4. Once confirmed, transfer them back to the greeter\n"
                    "Be attentive and ensure order accuracy before proceeding."
                ),
                fncs=[
                    self.update_order,
                    self.transfer_to_greeter,
                ],
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
                fncs=[
                    self.collect_name,
                    self.collect_phone,
                    self.transfer_to_greeter,
                ],
            ),
        }

        self._cur_spec = "Greeter"
        self._order: str | None = None
        self._customer_name: str | None = None
        self._customer_phone: str | None = None

    @property
    def spec(self) -> AgentSpec:
        return self._specs[self._cur_spec]

    def _transfer_to_spec(self, spec_name: str, call_ctx: AgentCallContext) -> None:
        agent = call_ctx.agent

        keep_last_n = 6
        prev_messages = agent.chat_ctx.messages.copy()
        while prev_messages and prev_messages[0].role in ["system", "tool"]:
            prev_messages.pop(0)
        prev_messages = prev_messages[-keep_last_n:]

        self._cur_spec = spec_name
        agent._fnc_ctx = self.spec.fnc_ctx
        agent._chat_ctx = self.spec.chat_ctx
        agent._chat_ctx.messages.extend(prev_messages)

        # use the new chat_ctx in the call_ctx
        call_ctx.chat_ctx.messages = agent.chat_ctx.messages.copy()
        logger.info(f"Transferred to {spec_name}")

    @llm.ai_callable()
    async def update_order(
        self,
        item: Annotated[
            str,
            llm.TypeInfo(
                description="The items of the full order, separated by commas"
            ),
        ],
    ):
        """Called when the user updates their order."""
        self._order = item
        logger.info("Updated order", extra={"order": item})
        return f"Updated order to {item}"

    @llm.ai_callable()
    async def collect_name(
        self, name: Annotated[str, llm.TypeInfo(description="The customer's name")]
    ):
        """Called when the user provides their name."""
        self._customer_name = name
        logger.info("Collected name", extra={"customer_name": name})
        return f"The name is updated to {name}"

    @llm.ai_callable()
    async def collect_phone(
        self,
        phone: Annotated[str, llm.TypeInfo(description="The customer's phone number")],
    ):
        """Called when the user provides their phone number."""
        # validate phone number
        phone = phone.strip().replace("-", "")
        if not phone.isdigit() or len(phone) != 10:
            return "The phone number is not valid, please try again."

        self._customer_phone = phone
        logger.info("Collected phone", extra={"customer_phone": phone})
        return f"The phone number is updated to {phone}"

    @llm.ai_callable()
    async def transfer_to_ordering(self):
        """Called to transfer the call to order taking."""
        call_ctx = AgentCallContext.get_current()
        self._transfer_to_spec("OrderTaking", call_ctx)
        return f"Transferred to order taking, the current order is {self._order}"

    @llm.ai_callable()
    async def transfer_to_info_collection(self):
        """Called to transfer the call to collect the customer's details."""
        call_ctx = AgentCallContext.get_current()
        self._transfer_to_spec("CustomerDetails", call_ctx)
        return (
            f"Transferred to collecting customer details, "
            f"the current collected name is {self._customer_name} "
            f"and phone number is {self._customer_phone}"
        )

    @llm.ai_callable()
    async def transfer_to_greeter(
        self,
        # summary: Annotated[
        #     str,
        #     llm.TypeInfo(
        #         description="The summary of conversations in the current stage"
        #     ),
        # ],
    ):
        """Called to transfer the call back to the greeter."""
        # message = f"Back to the greeter from {self._cur_spec}, the summary of conversations is {summary}"
        message = f"Back to the greeter from {self._cur_spec}. "
        if self._cur_spec == "OrderTaking":
            message += f"The current order is {self._order}"
        elif self._cur_spec == "CustomerDetails":
            message += (
                f"The current collected name is {self._customer_name} "
                f"and phone number is {self._customer_phone}"
            )
        logger.info("Back to greeter", extra={"summary": message})

        call_ctx = AgentCallContext.get_current()
        self._transfer_to_spec("Greeter", call_ctx)
        return message


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    menu = "Pizza, Salad, Ice Cream, Coffee"
    multi_stage_ctx = RestaurantBot(menu)

    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        fnc_ctx=multi_stage_ctx.spec.fnc_ctx,
        chat_ctx=multi_stage_ctx.spec.chat_ctx,
        max_nested_fnc_calls=2,  # may call functions in the transition function
    )

    @ctx.room.on("data_received")
    def on_data_received(packet: rtc.DataPacket):
        if packet.topic == "lk-chat-topic":
            data = json.loads(packet.data.decode("utf-8"))
            logger.info(f"Text input received: {data['message']}")

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
        f"Welcome to our restaurant! We offer {menu}. How may I assist you today?"
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )
