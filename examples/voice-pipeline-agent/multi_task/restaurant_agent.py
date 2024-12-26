import json
import logging
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
from livekit.agents.pipeline.agent_task import (
    AgentTask,
    AgentTaskOptions,
    _default_before_enter_cb,
)
from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.plugins import deepgram, openai, silero

load_dotenv()

logger = logging.getLogger("multi-task-agent")
logger.setLevel(logging.INFO)


user_data_template = {
    "order": [],
    "customer_name": None,
    "customer_phone": None,
    "checked_out": False,
    "expense": None,
}


async def before_enter_cb(
    agent: VoicePipelineAgent, task: AgentTask
) -> tuple[AgentTask, str]:
    task, message = await _default_before_enter_cb(agent, task)

    # additionally add the current user data to the message
    user_data = user_data_template.copy()
    user_data.update(agent.user_data)
    message += f" The current user data is {json.dumps(user_data)}"
    logger.info(message)
    return task, message


class Greeter(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            name="greeter",
            instructions=(
                "You are a friendly restaurant receptionist. Your tasks:\n"
                "1. Warmly greet the caller\n"
                f"2. Ask if they'd like to place an order. (menu: {menu})\n"
                "Transfer to:\n"
                "- order_taking: when ready to place order\n"
                "- customer_registration: only after order is complete\n"
                "- checkout: only after customer details are collected\n\n"
                "Important:\n"
                "- If a transfer function is unavailable, it means prerequisites aren't met\n"
                "- Guide the customer to complete previous steps first\n"
                "- If already checked out, start a new order\n\n"
                "For non-order inquiries, assist directly while maintaining a professional tone."
            ),
            functions=[self.start_new_order],
            options=AgentTaskOptions(before_enter_cb=before_enter_cb),
        )

    @llm.ai_callable()
    async def start_new_order(self) -> str:
        """Called to start a new order."""
        agent = AgentCallContext.get_current().agent
        agent.user_data.clear()
        logger.info("Started a new order")
        return "Started a new order"


"""
Another way to create a task

@llm.ai_callable()
async def start_new_order() -> str:
    ...

def can_enter_greeter(agent: VoicePipelineAgent) -> bool:
    return True

greeter = AgentTask(
    name="greeter",
    instructions="...",
    functions=[start_new_order],
    options=AgentTaskOptions(
        can_enter_cb=can_enter_greeter,
        before_enter_cb=before_enter_cb,
    ),
)
"""


class OrderTaking(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            name="order_taking",
            instructions=(
                "You are a professional order taker at a restaurant. Your tasks:\n"
                f"1. Take orders from our menu: {menu}\n"
                "2. Clarify special requests\n"
                "3. Confirm order accuracy\n\n"
                "Transfer to:\n"
                "- customer_registration: when order is confirmed\n"
                "- greeter: for general questions or starting over\n\n"
                "Important:\n"
                "- Use update_order function to save the order\n"
                "- Ensure order is complete before transferring to customer details\n"
                "- For non-order questions, transfer to greeter"
            ),
            functions=[self.update_order],
            options=AgentTaskOptions(
                can_enter_cb=self.can_enter,
                before_enter_cb=before_enter_cb,
            ),
        )

    def can_enter(self, agent: "VoicePipelineAgent") -> bool:
        checked_out = agent.user_data.get("checked_out", False)
        return not checked_out

    @llm.ai_callable()
    async def update_order(
        self,
        items: Annotated[
            list[str],
            llm.TypeInfo(description="The items of the full order"),
        ],
    ) -> str:
        """Called when the user updates their order."""

        agent = AgentCallContext.get_current().agent
        agent.user_data["order"] = items

        logger.info("Updated order", extra={"order": items})
        return f"Updated order to {items}"


class CustomerRegistration(AgentTask):
    def __init__(self):
        super().__init__(
            name="customer_registration",
            instructions=(
                "You are collecting customer information for their order. Your tasks:\n"
                "1. Get and confirm customer's name and comfirm the spelling\n"
                "2. Get phone number and verify it's correct\n"
                "3. Repeat both pieces of information back to ensure accuracy\n"
                "Transfer to:\n"
                "- checkout: when all details are confirmed\n"
                "- order_taking: to modify the order\n"
                "- greeter: for general questions\n\n"
                "Important:\n"
                "- Use collect_name and collect_phone functions to save details\n"
                "- Verify all information before proceeding to checkout\n"
                "- For non-detail questions, transfer to greeter"
            ),
            functions=[self.collect_name, self.collect_phone],
            options=AgentTaskOptions(
                can_enter_cb=self.can_enter,
                before_enter_cb=before_enter_cb,
            ),
        )

    def can_enter(self, agent: "VoicePipelineAgent") -> bool:
        checked_out = agent.user_data.get("checked_out", False)
        order = agent.user_data.get("order", None)
        return order and not checked_out

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
            name="checkout",
            instructions=(
                "You are a checkout agent at a restaurant. Your tasks:\n"
                f"1. Review order and prices ({menu})\n"
                "2. Calculate and confirm total\n"
                "3. Process checkout\n\n"
                "Transfer to:\n"
                "- order_taking: to modify order\n"
                "- customer_registration: to update information\n"
                "- greeter: after checkout or for general questions\n\n"
                "Important:\n"
                "- Use checkout function with final expense\n"
                "- After successful checkout, transfer to greeter\n"
                "- For non-checkout questions, transfer to greeter"
            ),
            functions=[self.checkout],
            options=AgentTaskOptions(
                can_enter_cb=self.can_enter,
                before_enter_cb=before_enter_cb,
            ),
        )

    def can_enter(self, agent: "VoicePipelineAgent") -> bool:
        checked_out = agent.user_data.get("checked_out", False)
        order = agent.user_data.get("order")
        customer_name = agent.user_data.get("customer_name")
        customer_phone = agent.user_data.get("customer_phone")

        return order and customer_name and customer_phone and not checked_out

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


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    chat_log_file = "multi_task_chat_log.txt"
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"

    # Set up chat logger
    chat_logger = logging.getLogger("chat_logger")
    chat_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(chat_log_file)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    chat_logger.addHandler(handler)

    agent_tasks = [
        Greeter(menu),
        OrderTaking(menu),
        CustomerRegistration(),
        Checkout(menu),
    ]

    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        initial_task=agent_tasks[0],
        available_tasks=agent_tasks,
        max_nested_fnc_calls=3,  # may call functions in the transition function
    )

    # read text input from the room for easy testing
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

    # write the chat log to a file
    @agent.on("user_speech_committed")
    @agent.on("agent_speech_interrupted")
    @agent.on("agent_speech_committed")
    def on_speech_committed(message: llm.ChatMessage):
        chat_logger.info(f"{message.role}: {message.content}")

    @agent.on("function_calls_collected")
    def on_function_calls_collected(calls: list[llm.FunctionCallInfo]):
        fnc_infos = [{fnc.function_info.name: fnc.arguments} for fnc in calls]
        chat_logger.info(f"fnc_calls_collected: {fnc_infos}")

    @agent.on("function_calls_finished")
    def on_function_calls_finished(calls: list[llm.CalledFunction]):
        called_infos = [{fnc.call_info.function_info.name: fnc.result} for fnc in calls]
        chat_logger.info(f"fnc_calls_finished: {called_infos}")

    # Start the assistant. This will automatically publish a microphone track and listen to the participant.
    agent.start(ctx.room, participant)
    await agent.say("Welcome to our restaurant! How may I assist you today?")


def prewarm_process(proc: JobProcess):
    # preload silero VAD in memory to speed up session start
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )
