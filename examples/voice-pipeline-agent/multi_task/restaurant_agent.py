import json
import logging
from typing import Annotated, Optional, TypedDict

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
from livekit.plugins import cartesia, deepgram, openai, silero

load_dotenv()

logger = logging.getLogger("multi-task-agent")
logger.setLevel(logging.INFO)


class UserData(TypedDict):
    customer_name: Optional[str]
    customer_phone: Optional[str]

    reservation_time: Optional[str]

    order: Optional[list[str]]
    customer_credit_card: Optional[str]
    customer_credit_card_expiry: Optional[str]
    customer_credit_card_cvv: Optional[str]
    expense: Optional[float]
    checked_out: Optional[bool]


def update_chat_ctx(task: AgentTask, chat_ctx: llm.ChatContext) -> AgentTask:
    last_chat_ctx = chat_ctx.truncate(keep_last_n=6)
    task.inject_chat_ctx(last_chat_ctx)
    return task


# some common functions
@llm.ai_callable()
async def update_name(
    name: Annotated[str, llm.TypeInfo(description="The customer's name")],
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    agent = AgentCallContext.get_current().agent
    user_data: UserData = agent.user_data
    user_data["customer_name"] = name
    return f"The name is updated to {name}"


@llm.ai_callable()
async def update_phone(
    phone: Annotated[str, llm.TypeInfo(description="The customer's phone number")],
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""

    agent = AgentCallContext.get_current().agent
    user_data: UserData = agent.user_data
    user_data["customer_phone"] = phone
    return f"The phone number is updated to {phone}"


@llm.ai_callable()
async def to_greeter() -> tuple[AgentTask, str]:
    """Called when user asks unrelated questions or requests other services."""
    agent = AgentCallContext.get_current().agent
    next_task = AgentTask.get_task(Greeter)
    return update_chat_ctx(next_task, agent.chat_ctx), f"User data: {agent.user_data}"


class Greeter(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist. The menu is: {menu}\n"
                "Your jobs are to greet the caller and guide them to the reservation, "
                "or order taking task based on the user's response."
            )
        )
        self.menu = menu

    @llm.ai_callable()
    async def to_reservation(self) -> tuple[AgentTask, str]:
        """Called when user wants to make a reservation."""
        agent = AgentCallContext.get_current().agent
        next_task = self.get_task(Reservation)
        return update_chat_ctx(
            next_task, agent.chat_ctx
        ), f"User info: {agent.user_data}"

    @llm.ai_callable()
    async def to_takeaway(self) -> tuple[AgentTask, str]:
        """Called when the user orders a takeaway"""
        agent = AgentCallContext.get_current().agent
        next_task = self.get_task(Takeaway)
        return update_chat_ctx(
            next_task, agent.chat_ctx
        ), f"User info: {agent.user_data}"

    @llm.ai_callable()
    def to_checkout(self) -> tuple[AgentTask, str]:
        """Called when the user wants to checkout the takeaway order."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        if user_data.get("checked_out"):
            return "You have already checked out. Please ask for other services."

        if not user_data.get("order"):
            next_task = self.get_task(Takeaway)
            message = "No order found. Please take an order first."
        else:
            next_task = self.get_task(Checkout)
            message = f"User order: {user_data['order']}. "

        message += f" User info: {user_data}"
        return update_chat_ctx(next_task, agent.chat_ctx), message


class Reservation(AgentTask):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a reservation agent at a restaurant. Your jobs are to record "
                "the customer's name, phone number, and reservation time. Then confirm "
                "the reservation details with the customer."
            ),
            functions=[update_name, update_phone, to_greeter],
        )

    @llm.ai_callable()
    async def update_reservation_time(
        self,
        time: Annotated[str, llm.TypeInfo(description="The reservation time")],
    ) -> str:
        """Called when the user provides their reservation time.
        Confirm the time with the user before calling the function."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        user_data["reservation_time"] = time
        return f"The reservation time is updated to {time}"

    @llm.ai_callable()
    async def confirm_reservation(self) -> str:
        """Called when the user confirms the reservation.
        Call this function to transfer to the next step."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        if not user_data.get("customer_name") or not user_data.get("customer_phone"):
            return "Please provide your name and phone number first."

        if not user_data.get("reservation_time"):
            return "Please provide reservation time first."

        next_task = self.get_task(Greeter)
        return update_chat_ctx(
            next_task, agent.chat_ctx
        ), f"Reservation confirmed. User data: {user_data}"


class Takeaway(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "You are a professional order taker at a restaurant Our menu is: "
                f"{menu}. Your jobs are to show the menu and take orders from the "
                "customer. Clarify special requests and confirm the order with the "
                "customer."
            ),
            functions=[to_greeter],
        )

    @llm.ai_callable()
    async def update_order(
        self,
        items: Annotated[
            list[str], llm.TypeInfo(description="The items of the full order")
        ],
    ) -> str:
        """Called when the user updates their order."""

        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        user_data["order"] = items
        return f"Updated order to {items}"

    @llm.ai_callable()
    async def to_checkout(self) -> tuple[AgentTask, str]:
        """Called when the user confirms the order. Call this function to transfer to the checkout step.
        Double check the order with the user before calling the function."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        if not user_data.get("order"):
            return "No order found. Please take an order first."

        next_task = self.get_task(Checkout)
        return update_chat_ctx(next_task, agent.chat_ctx), f"User info: {user_data}"


class Checkout(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "You are a professional checkout agent at a restaurant. The menu is: "
                f"{menu}. Your are responsible for calculating the expense of the "
                "order and collecting customer's name, phone number and credit card "
                "information, including the card number, expiry date, and CVV."
            ),
            functions=[update_name, update_phone, to_greeter],
        )

    @llm.ai_callable()
    async def confirm_expense(
        self,
        expense: Annotated[float, llm.TypeInfo(description="The expense of the order")],
    ) -> str:
        """Called when the user confirms the expense."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        user_data["expense"] = expense
        return f"The expense is confirmed to be {expense}"

    @llm.ai_callable()
    async def update_credit_card(
        self,
        number: Annotated[str, llm.TypeInfo(description="The credit card number")],
        expiry: Annotated[
            str, llm.TypeInfo(description="The expiry date of the credit card")
        ],
        cvv: Annotated[str, llm.TypeInfo(description="The CVV of the credit card")],
    ) -> str:
        """Called when the user provides their credit card number, expiry date, and CVV.
        Confirm the spelling with the user before calling the function."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        user_data["customer_credit_card"] = number
        user_data["customer_credit_card_expiry"] = expiry
        user_data["customer_credit_card_cvv"] = cvv
        return f"The credit card number is updated to {number}"

    @llm.ai_callable()
    async def confirm_checkout(self) -> str:
        """Called when the user confirms the checkout.
        Double check the information with the user before calling the function."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        if not user_data.get("expense"):
            return "Please confirm the expense first."

        if (
            not user_data.get("customer_credit_card")
            or not user_data.get("customer_credit_card_expiry")
            or not user_data.get("customer_credit_card_cvv")
        ):
            return "Please provide the credit card information first."

        user_data["checked_out"] = True
        next_task = self.get_task(Greeter)
        return update_chat_ctx(
            next_task, agent.chat_ctx
        ), f"User checked out. User info: {user_data}"


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # create tasks
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    greeter = AgentTask.register_task(Greeter(menu))
    AgentTask.register_task(Reservation())
    AgentTask.register_task(Takeaway(menu))
    AgentTask.register_task(Checkout(menu))

    # Set up chat logger
    chat_log_file = "restaurant_agent.log"
    chat_logger = logging.getLogger("chat_logger")
    chat_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(chat_log_file)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    chat_logger.addHandler(handler)

    participant = await ctx.wait_for_participant()
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        initial_task=greeter,
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
