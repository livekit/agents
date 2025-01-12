import json
import logging
from typing import Annotated, Optional, TypedDict, TypeVar

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
    AgentInlineTask,
    AgentTask,
    ResultNotSet,
    TaskFailed,
)
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


T = TypeVar("T", bound=AgentTask)


def update_chat_ctx(task: T, chat_ctx: llm.ChatContext) -> T:
    last_chat_ctx = chat_ctx.truncate(keep_last_n=6)
    task.inject_chat_ctx(last_chat_ctx)
    return task


class GetName(AgentInlineTask):
    def __init__(self, user_data: UserData | None = None):
        instructions = "Ask for and collect the customer's name. Please verify the spelling before proceeding."
        if user_data:
            instructions += f"\nPreviously collected user data: {user_data}"
        super().__init__(instructions=instructions)

    @llm.ai_callable()
    async def set_name(
        self, name: Annotated[str, llm.TypeInfo(description="The user's name")]
    ) -> str:
        """Called when the user provides their name."""
        self._result = name
        return f"The name is updated to {name}"


class GetPhoneNumber(AgentInlineTask):
    def __init__(self, user_data: UserData | None = None):
        instructions = "Ask for and collect a valid 10-digit phone number. Please verify the number before proceeding."
        if user_data:
            instructions += f"\nPreviously collected user data: {user_data}"
        super().__init__(instructions=instructions)

    @llm.ai_callable()
    async def set_phone_number(
        self,
        phone_number: Annotated[
            str, llm.TypeInfo(description="The user's phone number")
        ],
    ) -> str:
        """Called when the user provides their phone number."""
        # validate the phone number
        phone_number = phone_number.strip().replace("-", "")
        if not phone_number.isdigit() or len(phone_number) != 10:
            return (
                "The phone number is not valid. Please provide a 10-digit phone number."
            )

        self._result = phone_number
        return f"The phone number is updated to {phone_number}"


class GetReservationTime(AgentInlineTask):
    def __init__(self, user_data: UserData | None = None):
        instructions = "Ask for the desired reservation time and confirm the timing with the customer."
        if user_data:
            instructions += f"\nPreviously collected user data: {user_data}"
        super().__init__(instructions=instructions)

    @llm.ai_callable()
    async def set_reservation_time(
        self, time: Annotated[str, llm.TypeInfo(description="The reservation time")]
    ) -> str:
        """Called when the user provides their reservation time."""
        self._result = time
        return f"The reservation time is updated to {time}"


class TakeOrder(AgentInlineTask):
    def __init__(self, menu: str, user_data: UserData | None = None):
        instructions = (
            "You are a professional restaurant server. Here's our menu: "
            f"{menu}. Take the customer's order, clarify any special requests, "
            "and confirm the complete order before proceeding."
        )
        if user_data:
            instructions += f"\nPreviously collected user data: {user_data}"
        super().__init__(instructions=instructions)

    @llm.ai_callable()
    async def update_order(
        self,
        items: Annotated[
            list[str], llm.TypeInfo(description="The items of the full order")
        ],
    ) -> str:
        """Called when the user updates their order."""
        self._result = items
        if not items:
            return "All items are removed from the order."

        return f"Updated order to {items}"


class GetCreditCard(AgentInlineTask):
    def __init__(self, user_data: UserData | None = None):
        instructions = "Collect the customer's payment information: card number, expiration date (MM/YY), and CVV."
        if user_data:
            instructions += f"\nPreviously collected user data: {user_data}"
        super().__init__(instructions=instructions)

    @llm.ai_callable()
    async def set_credit_card(
        self,
        number: Annotated[str, llm.TypeInfo(description="The credit card number")],
        expiry: Annotated[
            str,
            llm.TypeInfo(
                description="The expiry date of the credit card, in MM/YY format"
            ),
        ],
        cvv: Annotated[str, llm.TypeInfo(description="The CVV of the credit card")],
    ) -> str:
        """Called when the user provides their credit card information."""

        # validate the credit card information
        if not cvv.isdigit() or len(cvv) != 3:
            return "The CVV is not valid. Please provide a 3-digit CVV."

        # validate the expiry date
        month, year = expiry.split("/")
        if (
            not month.isdigit()
            or not year.isdigit()
            or len(month) != 2
            or len(year) != 2
        ):
            return "The expiry date is not valid."

        self._result = {
            "customer_credit_card": number,
            "customer_credit_card_expiry": expiry,
            "customer_credit_card_cvv": cvv,
        }
        return f"The credit card information is updated to {self._result}"


class Greeter(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                f"You are a friendly restaurant host. Our menu: {menu}\n"
                "Welcome customers and guide them to either make a reservation "
                "or place an order based on their preference."
            )
        )
        self.menu = menu

    @llm.ai_callable()
    async def make_reservation(self) -> str:
        """Called when the user want to make a reservation."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data

        try:
            reservation_time = await update_chat_ctx(
                GetReservationTime(user_data), agent.chat_ctx
            ).run()
            user_data["reservation_time"] = reservation_time

            name = await update_chat_ctx(GetName(user_data), agent.chat_ctx).run()
            user_data["customer_name"] = name

            phone = await update_chat_ctx(
                GetPhoneNumber(user_data), agent.chat_ctx
            ).run()
            user_data["customer_phone"] = phone
        except TaskFailed as e:
            return f"Task failed: {e}"
        except ResultNotSet:
            return f"Failed to collect user data, the collected data is {user_data}"

        return f"Reservation successful. Updated user data: {user_data}"

    @llm.ai_callable()
    async def take_order(self) -> str:
        """Called when the user wants to take an order."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data

        try:
            order = await update_chat_ctx(
                TakeOrder(self.menu, user_data), agent.chat_ctx
            ).run()
            user_data["order"] = order
        except TaskFailed as e:
            return f"Task failed: {e}"
        except ResultNotSet:
            return f"Failed to collect the order, the collected data is {user_data}"
        return f"Order successful. Updated user data: {user_data}"

    @llm.ai_callable()
    async def checkout(
        self,
        expense: Annotated[float, llm.TypeInfo(description="The expense of the order")],
    ) -> str:
        """Called when the user confirms the expense of the order and want to checkout."""
        agent = AgentCallContext.get_current().agent
        user_data: UserData = agent.user_data
        user_data["expense"] = expense

        try:
            name = await update_chat_ctx(GetName(user_data), agent.chat_ctx).run()
            user_data["customer_name"] = name

            phone = await update_chat_ctx(
                GetPhoneNumber(user_data), agent.chat_ctx
            ).run()
            user_data["customer_phone"] = phone

            credit_card = await update_chat_ctx(
                GetCreditCard(user_data), agent.chat_ctx
            ).run()
            user_data.update(credit_card)
        except TaskFailed as e:
            return f"Task failed: {e}"
        except ResultNotSet:
            return f"Failed to collect user data, the collected data is {user_data}"

        user_data["checked_out"] = True

        return f"Updated user data: {user_data}. User checked out."


@llm.ai_callable()
async def to_greeter() -> tuple[AgentTask, str]:
    """Called when user asks unrelated questions or requests other services."""
    agent = AgentCallContext.get_current().agent
    next_task = AgentTask.get_task(Greeter)
    return update_chat_ctx(next_task, agent.chat_ctx), f"User data: {agent.user_data}"


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # create tasks
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    greeter = AgentTask.register_task(Greeter(menu))

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
