import logging
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask, CallContext, VoiceAgent
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai
from pydantic import Field

# from livekit.plugins import noise_cancellation

logger = logging.getLogger("restaurant-example")
logger.setLevel(logging.INFO)

load_dotenv()


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

    tasks: dict[str, AgentTask]


CallContext_T = CallContext[UserData]


# common functions
@ai_function()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")], context: CallContext_T
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata["customer_name"] = name
    return f"The name is updated to {name}"


@ai_function()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")], context: CallContext_T
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata["customer_phone"] = phone
    return f"The phone number is updated to {phone}"


@ai_function()
async def to_greeter(context: CallContext_T) -> tuple[AgentTask, str]:
    """Called when user asks any unrelated questions or requests any other services not in your job description."""
    userdata = context.userdata
    next_task = userdata["tasks"]["greeter"]

    # TODO: update chat context
    return next_task, "Transferring to greeter."


class Greeter(AgentTask):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist. The menu is: {menu}\n"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent. "
            ),
        )
        self.menu = menu

    async def on_enter(self) -> None:
        logger.info("enter greeter now")
        # self.agent.generate_reply()

    @ai_function
    async def to_reservation(self, context: CallContext_T) -> tuple[AgentTask, str]:
        userdata = context.userdata

        # TODO: update chat context
        return userdata["tasks"]["reservation"], "Transferring to reservation."

    @ai_function
    async def to_takeaway(self, context: CallContext_T) -> tuple[AgentTask, str]:
        userdata = context.userdata

        # TODO: update chat context
        return userdata["tasks"]["takeaway"], "Transferring to takeaway."


class Reservation(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a reservation agent at a restaurant. Your jobs are to ask for "
            "the reservation time, then customer's name, and phone number. Then "
            "confirm the reservation details with the customer.",
            ai_functions=[update_name, update_phone, to_greeter],
        )

    async def on_enter(self) -> None:
        logger.info("enter reservation now")
        self.agent.generate_reply()

    @ai_function()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: CallContext,
    ) -> str:
        userdata = context.userdata
        userdata["reservation_time"] = time
        return f"The reservation time is updated to {time}"

    @ai_function
    async def confirm_reservation(self, context: CallContext_T) -> str | tuple[AgentTask, str]:
        userdata = context.userdata
        if not userdata.get("customer_name") or not userdata.get("customer_phone"):
            return "Please provide your name and phone number first."

        if not userdata.get("reservation_time"):
            return "Please provide reservation time first."

        return await to_greeter(context)


class Takeaway(AgentTask):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=f"Our menu is: {menu}. Your jobs are to record the order from the "
            "customer. Clarify special requests and confirm the order with the "
            "customer.",
            ai_functions=[to_greeter],
        )

    async def on_enter(self) -> None:
        logger.info("enter takeaway now")
        userdata: UserData = self.agent.userdata
        self.agent.generate_reply(user_input=f"Current order is {userdata.get('order', [])}")

    @ai_function
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: CallContext_T,
    ) -> str:
        userdata = context.userdata
        userdata["order"] = items
        return f"The order is updated to {items}"

    @ai_function
    async def to_checkout(self, context: CallContext_T) -> str | tuple[AgentTask, str]:
        userdata = context.userdata
        if not userdata.get("order"):
            return "No takeaway order found. Please make an order first."

        return userdata["tasks"]["checkout"], "Transferring to checkout."


class Checkout(AgentTask):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                "You are a professional checkout agent at a restaurant. The menu is: "
                f"{menu}. Your are responsible for confirming the expense of the "
                "order and then collecting customer's name, phone number and credit card "
                "information, including the card number, expiry date, and CVV step by step."
            ),
            ai_functions=[update_name, update_phone, to_greeter],
        )

    async def on_enter(self) -> None:
        logger.info("enter checkout now")
        userdata: UserData = self.agent.userdata
        # TODO: how to add customer name, phone number, etc. to the chat context if they were already collected?
        self.agent.generate_reply(user_input=f"The order is {userdata.get('order', [])}")

    @ai_function
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: CallContext_T,
    ) -> str:
        userdata = context.userdata
        userdata["expense"] = expense
        return f"The expense is confirmed to be {expense}"

    @ai_function
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: CallContext_T,
    ) -> str:
        userdata = context.userdata
        userdata["customer_credit_card"] = number
        userdata["customer_credit_card_expiry"] = expiry
        userdata["customer_credit_card_cvv"] = cvv
        return f"The credit card number is updated to {number}"

    @ai_function
    async def confirm_checkout(self, context: CallContext_T) -> str | tuple[AgentTask, str]:
        userdata = context.userdata
        if not userdata.get("expense"):
            return "Please confirm the expense first."

        if (
            not userdata.get("customer_credit_card")
            or not userdata.get("customer_credit_card_expiry")
            or not userdata.get("customer_credit_card_cvv")
        ):
            return "Please provide the credit card information first."

        userdata["checked_out"] = True
        return await to_greeter(context)

    @ai_function
    async def to_takeaway(self, context: CallContext_T) -> tuple[AgentTask, str]:
        userdata = context.userdata
        return userdata["tasks"]["takeaway"], "Transferring to takeaway."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    userdata = UserData()
    userdata["tasks"] = {
        "greeter": Greeter(menu),
        "reservation": Reservation(),
        "takeaway": Takeaway(menu),
        "checkout": Checkout(menu),
    }

    agent = VoiceAgent[UserData](
        task=userdata["tasks"]["greeter"],
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
    )

    await agent.start(
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await agent.say("Welcome to our restaurant! How may I assist you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
