import json
import logging
from typing import Annotated, Self

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
from livekit.agents.pipeline.agent_task import AgentTask
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.plugins import deepgram, openai, silero

load_dotenv()

logger = logging.getLogger("multi-stage-agent")
logger.setLevel(logging.INFO)


def get_last_n_messages(
    messages: list[llm.ChatMessage], n: int
) -> list[llm.ChatMessage]:
    collected_messages = messages.copy()[-n:]
    while collected_messages and collected_messages[0].role in ["system", "tool"]:
        collected_messages.pop(0)
    return collected_messages


def _transfer_to(task: AgentTask, message: str | None = None) -> tuple[AgentTask, str]:
    agent = AgentCallContext.get_current().agent

    # keep the last n messages for the next stage
    keep_last_n = 6
    task.chat_ctx.messages.extend(
        get_last_n_messages(agent.chat_ctx.messages, keep_last_n)
    )

    message = (
        message or f"Transferred from {agent.current_agent_task.name} to {task.name}"
    )
    logger.info(message)
    return task, message


class Greeter(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "You are a professional restaurant receptionist handling incoming calls. "
                "Warmly greet the caller and ask if they would like to place an order. "
                f"Available menu items: {menu}. "
                "Maintain a friendly and professional tone throughout the conversation.\n"
                "Guide the conversation as follows: order taking, customer details, checkout. "
                "Use the functions to transfer the call to OrderTaking, CustomerDetails, or Checkout. "
                "For any other inquiries, assist them directly."
            ),
        )

    def can_enter(self, agent: "VoicePipelineAgent") -> bool:
        return True

    @llm.ai_callable(name="enter_greeter")
    async def enter(self) -> tuple[Self, str]:
        """Called to transfer to the greeter."""

        agent = AgentCallContext.get_current().agent
        curr_task = agent.current_agent_task

        # return the collected information to the greeter
        message = f"Transferred from {curr_task.name} to {self.name}. "
        if isinstance(curr_task, OrderTaking):
            message += f"The current order is {agent.user_data.get('order', 'empty')}"
        elif isinstance(curr_task, CustomerDetails):
            message += (
                f"The customer name is {agent.user_data.get('customer_name', 'unknown')}, "
                f"phone number is {agent.user_data.get('customer_phone', 'unknown')}"
            )

        return _transfer_to(self, message)


class OrderTaking(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "You are a professional order taker at a restaurant. "
                "Guide the customer through their order with these steps:\n"
                f"1. Take their order selections one at a time from our menu: {menu}\n"
                "2. Clarify any special requests or modifications\n"
                "3. Repeat back the complete order to confirm accuracy\n"
                "4. Once confirmed, transfer them to collect customer details.\n"
                "Be attentive and ensure order accuracy before proceeding."
                "Use the functions to transfer the call to the next step."
            ),
            functions=[self.update_order],
        )

    def can_enter(self, agent: "VoicePipelineAgent") -> bool:
        checked_out = agent.user_data.get("checked_out", False)
        return not checked_out

    @llm.ai_callable(name="enter_order_taking")
    async def enter(self) -> tuple[Self, str]:
        """Called to transfer to the order taking."""
        return _transfer_to(self)

    @llm.ai_callable()
    async def update_order(
        self,
        item: Annotated[
            str,
            llm.TypeInfo(
                description="The items of the full order, separated by commas"
            ),
        ],
    ) -> str:
        """Called when the user updates their order."""

        agent = AgentCallContext.get_current().agent
        agent.user_data["order"] = item

        logger.info("Updated order", extra={"order": item})
        return f"Updated order to {item}"


class CustomerDetails(AgentTask):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are collecting essential customer information for their order. "
                "Follow these steps carefully:\n"
                "1. Ask for the customer's name and confirm the spelling\n"
                "2. Request their phone number and verify it's correct\n"
                "3. Repeat both pieces of information back to ensure accuracy\n"
                "4. Once confirmed, transfer to checkout.\n"
                "Handle personal information professionally and courteously."
                "Use the functions to transfer the call to the next step."
            ),
            functions=[self.collect_name, self.collect_phone],
        )

    def can_enter(self, agent: "VoicePipelineAgent") -> bool:
        checked_out = agent.user_data.get("checked_out", False)
        order = agent.user_data.get("order", None)
        return order and not checked_out

    @llm.ai_callable(name="enter_customer_details")
    async def enter(self) -> tuple[Self, str]:
        """Called to transfer to the customer details."""
        return _transfer_to(self)

    @llm.ai_callable()
    async def collect_name(
        self, name: Annotated[str, llm.TypeInfo(description="The customer's name")]
    ) -> str:
        """Called when the user provides their name."""
        agent = AgentCallContext.get_current().agent
        agent.user_data["customer_name"] = name

        logger.info("Collected name", extra={"customer_name": name})
        return f"The name is updated to {name}"

    @llm.ai_callable()
    async def collect_phone(
        self,
        phone: Annotated[str, llm.TypeInfo(description="The customer's phone number")],
    ) -> str:
        """Called when the user provides their phone number."""
        agent = AgentCallContext.get_current().agent
        agent.user_data["customer_phone"] = phone

        logger.info("Collected phone", extra={"customer_phone": phone})
        return f"The phone number is updated to {phone}"


class Checkout(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "You are a checkout agent. Ask the customer if they want to checkout. "
                f"The menu items and prices are: {menu}. "
                "If they confirm, call the checkout function and transfer them back to the greeter."
            ),
            functions=[self.checkout],
        )

    def can_enter(self, agent: "VoicePipelineAgent") -> bool:
        checked_out = agent.user_data.get("checked_out", False)
        order = agent.user_data.get("order", None)
        customer_name = agent.user_data.get("customer_name", None)
        customer_phone = agent.user_data.get("customer_phone", None)

        return order and customer_name and customer_phone and not checked_out

    @llm.ai_callable(name="enter_checkout")
    async def enter(self) -> tuple[Self, str]:
        """Called to transfer to the checkout."""
        agent = AgentCallContext.get_current().agent
        message = f"Transferred from {agent.current_agent_task.name} to {self.name}. "
        message += f"The current order is {agent.user_data.get('order', 'empty')}. "
        message += f"The customer name is {agent.user_data.get('customer_name', 'unknown')}, "
        message += f"phone number is {agent.user_data.get('customer_phone', 'unknown')}"
        return _transfer_to(self, message)

    @llm.ai_callable()
    async def checkout(
        self,
        expense: Annotated[float, llm.TypeInfo(description="The expense of the order")],
    ) -> str:
        """Called when the user confirms the checkout."""
        agent = AgentCallContext.get_current().agent
        agent.user_data["checked_out"] = True
        agent.user_data["expense"] = expense
        logger.info("Checked out", extra=agent.user_data)
        return "Checked out"


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    chat_log_file = "multi_stage_chat_log.txt"
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    agent_tasks = [Greeter(menu), OrderTaking(menu), CustomerDetails(), Checkout(menu)]

    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        initial_task=agent_tasks[0],
        available_tasks=agent_tasks,
        max_nested_fnc_calls=2,  # may call functions in the transition function
    )

    # For testing with text input
    @ctx.room.on("data_received")
    def on_data_received(packet: rtc.DataPacket):
        if packet.topic == "lk-chat-topic":
            data = json.loads(packet.data.decode("utf-8"))
            logger.debug("Text input received", extra={"message": data["message"]})

            agent._human_input.emit(
                "final_transcript",
                SpeechEvent(
                    type=SpeechEventType.END_OF_SPEECH,
                    alternatives=[SpeechData(language="en", text=data["message"])],
                ),
            )

    @agent.on("user_speech_committed")
    @agent.on("agent_speech_interrupted")
    @agent.on("agent_speech_committed")
    def on_speech_committed(message: llm.ChatMessage):
        with open(chat_log_file, "a") as f:
            f.write(f"{message.role}: {message.content}\n")

    @agent.on("function_calls_collected")
    def on_function_calls_collected(calls: list[llm.FunctionCallInfo]):
        fnc_infos = [{fnc.function_info.name: fnc.arguments} for fnc in calls]
        with open(chat_log_file, "a") as f:
            f.write(f"fnc_calls_collected: {fnc_infos}\n")

    @agent.on("function_calls_finished")
    def on_function_calls_finished(calls: list[llm.CalledFunction]):
        called_infos = [{fnc.call_info.function_info.name: fnc.result} for fnc in calls]
        with open(chat_log_file, "a") as f:
            f.write(f"fnc_calls_finished: {called_infos}\n")

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
