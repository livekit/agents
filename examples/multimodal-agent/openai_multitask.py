from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    multimodal,
)
from livekit.agents.pipeline import AgentTask
from livekit.plugins import openai
from livekit.plugins.openai.realtime import RealtimeCallContext

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


def update_context(task: AgentTask, chat_ctx: llm.ChatContext) -> None:
    # last_chat_ctx = chat_ctx.truncate(keep_last_n=4, keep_tool_calls=False)
    # task.inject_chat_ctx(last_chat_ctx)
    pass


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


# some common functions
@llm.ai_callable()
async def update_name(
    name: Annotated[str, llm.TypeInfo(description="The customer's name")],
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    session = RealtimeCallContext.get_current().session
    user_data: UserData = session.user_data
    user_data["customer_name"] = name
    return f"The name is updated to {name}"


@llm.ai_callable()
async def update_phone(
    phone: Annotated[str, llm.TypeInfo(description="The customer's phone number")],
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""

    session = RealtimeCallContext.get_current().session
    user_data: UserData = session.user_data
    user_data["customer_phone"] = phone
    return f"The phone number is updated to {phone}"


@llm.ai_callable()
async def to_greeter() -> tuple[AgentTask, str]:
    """Called when user asks any unrelated questions or requests any other services not in your job description."""
    session = RealtimeCallContext.get_current().session
    next_task = AgentTask.get_task("greeter")
    update_context(next_task, session.chat_ctx_copy())
    return next_task, f"User data: {session.user_data}"


class Greeter(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist. The menu is: {menu}\n"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent."
            )
        )
        self.menu = menu

    @llm.ai_callable()
    async def to_reservation(self) -> tuple[AgentTask, str]:
        """Called when user wants to make a reservation. This function handles transitioning to the reservation agent
        who will collect the necessary details like reservation time, customer name and phone number."""
        session = RealtimeCallContext.get_current().session
        next_task = AgentTask.get_task("reservation")
        update_context(next_task, session.chat_ctx_copy())
        return next_task, f"User info: {session.user_data}"

    @llm.ai_callable()
    async def to_takeaway(self) -> tuple[AgentTask, str]:
        """Called when the user wants to place a takeaway order. This includes handling orders for pickup,
        delivery, or when the user wants to proceed to checkout with their existing order."""
        session = RealtimeCallContext.get_current().session
        next_task = AgentTask.get_task("takeaway")
        update_context(next_task, session.chat_ctx_copy())
        return next_task, f"User info: {session.user_data}"


class Reservation(AgentTask):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a reservation agent at a restaurant. Your jobs are to ask for "
                "the reservation time, then customer's name, and phone number. Then "
                "confirm the reservation details with the customer."
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
        session = RealtimeCallContext.get_current().session
        user_data: UserData = session.user_data
        user_data["reservation_time"] = time
        return f"The reservation time is updated to {time}"

    @llm.ai_callable()
    async def confirm_reservation(self) -> str:
        """Called when the user confirms the reservation.
        Call this function to transfer to the next step."""
        session = RealtimeCallContext.get_current().session
        user_data: UserData = session.user_data
        if not user_data.get("customer_name") or not user_data.get("customer_phone"):
            return "Please provide your name and phone number first."

        if not user_data.get("reservation_time"):
            return "Please provide reservation time first."

        next_task = AgentTask.get_task("greeter")
        update_context(next_task, session.chat_ctx_copy())
        return next_task, f"Reservation confirmed. User data: {user_data}"


class Takeaway(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                f"Our menu is: {menu}. Your jobs are to record the order from the "
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
        """Called when the user create or update their order."""
        session = RealtimeCallContext.get_current().session
        user_data: UserData = session.user_data
        user_data["order"] = items
        return f"Updated order to {items}"

    @llm.ai_callable()
    async def to_checkout(self) -> tuple[AgentTask, str]:
        """Called when the user confirms the order. Call this function to transfer to the checkout step.
        Double check the order with the user before calling the function."""
        session = RealtimeCallContext.get_current().session
        user_data: UserData = session.user_data
        if not user_data.get("order"):
            return "No takeaway order found. Please make an order first."

        next_task = AgentTask.get_task("checkout")
        update_context(next_task, session.chat_ctx_copy())
        return next_task, f"User info: {user_data}"


class Checkout(AgentTask):
    def __init__(self, menu: str):
        super().__init__(
            instructions=(
                "You are a professional checkout agent at a restaurant. The menu is: "
                f"{menu}. Your are responsible for confirming the expense of the "
                "order and then collecting customer's name, phone number and credit card "
                "information, including the card number, expiry date, and CVV step by step."
            ),
            functions=[update_name, update_phone, to_greeter],
        )

    @llm.ai_callable()
    async def confirm_expense(
        self,
        expense: Annotated[float, llm.TypeInfo(description="The expense of the order")],
    ) -> str:
        """Called when the user confirms the expense."""
        session = RealtimeCallContext.get_current().session
        user_data: UserData = session.user_data
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
        session = RealtimeCallContext.get_current().session
        user_data: UserData = session.user_data
        user_data["customer_credit_card"] = number
        user_data["customer_credit_card_expiry"] = expiry
        user_data["customer_credit_card_cvv"] = cvv
        return f"The credit card number is updated to {number}"

    @llm.ai_callable()
    async def confirm_checkout(self) -> str:
        """Called when the user confirms the checkout.
        Double check the information with the user before calling the function."""
        session = RealtimeCallContext.get_current().session
        user_data: UserData = session.user_data
        if not user_data.get("expense"):
            return "Please confirm the expense first."

        if (
            not user_data.get("customer_credit_card")
            or not user_data.get("customer_credit_card_expiry")
            or not user_data.get("customer_credit_card_cvv")
        ):
            return "Please provide the credit card information first."

        user_data["checked_out"] = True
        next_task = AgentTask.get_task("greeter")
        update_context(next_task, session.chat_ctx_copy())
        return next_task, f"User checked out. User info: {user_data}"

    @llm.ai_callable()
    async def to_takeaway(self) -> tuple[AgentTask, str]:
        """Called when the user wants to update their order."""
        session = RealtimeCallContext.get_current().session
        next_task = AgentTask.get_task("takeaway")
        update_context(next_task, session.chat_ctx_copy())
        return next_task, f"User info: {session.user_data}"


async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    # create tasks
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    greeter = AgentTask.register_task(Greeter(menu), "greeter")
    AgentTask.register_task(Reservation(), "reservation")
    AgentTask.register_task(Takeaway(menu), "takeaway")
    AgentTask.register_task(Checkout(menu), "checkout")

    agent = multimodal.MultimodalAgent(
        model=openai.realtime.RealtimeModel(
            voice="alloy",
            temperature=0.8,
            instructions=greeter.instructions,
            turn_detection=openai.realtime.ServerVadOptions(
                threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
            ),
        ),
        initial_task=greeter,
    )
    agent.start(ctx.room, participant)

    await asyncio.sleep(1)
    session: openai.realtime.RealtimeSession = agent._session
    session.response.create()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
