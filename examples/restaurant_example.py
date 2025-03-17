import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, CallContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai

# from livekit.plugins import noise_cancellation

logger = logging.getLogger("restaurant-example")
logger.setLevel(logging.INFO)

load_dotenv()


@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None

    reservation_time: Optional[str] = None

    order: Optional[list[str]] = None

    customer_credit_card: Optional[str] = None
    customer_credit_card_expiry: Optional[str] = None
    customer_credit_card_cvv: Optional[str] = None

    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    agents: dict[str, Agent] = field(default_factory=dict)

    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "credit_card": {
                "number": self.customer_credit_card or "unknown",
                "expiry": self.customer_credit_card_expiry or "unknown",
                "cvv": self.customer_credit_card_cvv or "unknown",
            }
            if self.customer_credit_card
            else None,
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        # summarize in yaml performs better than json
        return yaml.dump(data)


CallContext_T = CallContext[UserData]


# common functions


@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")], context: CallContext_T
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")], context: CallContext_T
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"


@function_tool()
async def to_greeter(context: CallContext_T) -> Agent:
    """Called when user asks any unrelated questions or requests any other services not in your job description."""
    curr_agent: BaseAgent = context.session.current_agent
    return curr_agent._transfer_to_agent("greeter", context)


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        logger.info(f"entering task {self.__class__.__name__}")
        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx
        chat_ctx.add_message(
            role="system",
            content=f"Current user data is {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply()

    async def _transfer_to_agent(self, name: str, context: CallContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]

        # copy part of chat history from current task to next task
        items_copy = self._truncate_chat_ctx(current_agent.chat_ctx.items)
        existing_ids = {item.id for item in next_agent.chat_ctx.items}
        items_copy = [item for item in items_copy if item.id not in existing_ids]

        chat_ctx = next_agent.chat_ctx
        chat_ctx.items.extend(items_copy)
        await next_agent.update_chat_ctx(chat_ctx)

        return next_agent, f"Transferring to {name}."

    def _truncate_chat_ctx(
        self,
        items: list[llm.ChatItem],
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list[llm.ChatItem]:
        """Truncate the chat context to keep the last n messages."""

        def _valid_item(item: llm.ChatItem) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in ["function_call", "function_call_output"]:
                return False
            return True

        new_items: list[llm.ChatItem] = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        # drop the function_call_output at the beginning
        while new_items and new_items[0].type == "function_call_output":
            new_items.pop(0)

        return new_items


class Greeter(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist. The menu is: {menu}\n"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent using tools."
            ),
            # llm=openai.LLM(model="gpt-4o-mini", parallel_tool_calls=False),
        )
        self.menu = menu

    @function_tool()
    async def to_reservation(self, context: CallContext_T) -> Agent:
        """Called when user wants to make a reservation.
        This function handles transitioning to the reservation agent
        who will collect the necessary details like reservation time,
        customer name and phone number."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: CallContext_T) -> Agent:
        """Called when the user wants to place a takeaway order.
        This includes handling orders for pickup, delivery, or when the user wants to
        proceed to checkout with their existing order."""
        return await self._transfer_to_agent("takeaway", context)


class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a reservation agent at a restaurant. Your jobs are to ask for "
            "the reservation time, then customer's name, and phone number. Then "
            "confirm the reservation details with the customer.",
            tools=[update_name, update_phone, to_greeter],
        )

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: CallContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.reservation_time = time
        return f"The reservation time is updated to {time}"

    @function_tool()
    async def confirm_reservation(self, context: CallContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."

        if not userdata.reservation_time:
            return "Please provide reservation time first."

        return await self._transfer_to_agent("greeter", context)


class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=f"Our menu is: {menu}. Your jobs are to record the order from the "
            "customer. Clarify special requests and confirm the order with the "
            "customer.",
            tools=[to_greeter],
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: CallContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.order = items
        return f"The order is updated to {items}"

    @function_tool()
    async def to_checkout(self, context: CallContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."

        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                "You are a professional checkout agent at a restaurant. The menu is: "
                f"{menu}. Your are responsible for confirming the expense of the "
                "order and then collecting customer's name, phone number and credit card "
                "information, including the card number, expiry date, and CVV step by step."
            ),
            tools=[update_name, update_phone, to_greeter],
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: CallContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: CallContext_T,
    ) -> str:
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card number is updated to {number}"

    @function_tool()
    async def confirm_checkout(self, context: CallContext_T) -> str | tuple[Agent, str]:
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."

        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."

        userdata.checked_out = True
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: CallContext_T) -> tuple[Agent, str]:
        return await self._transfer_to_agent("takeaway", context)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(menu),
            "reservation": Reservation(),
            "takeaway": Takeaway(menu),
            "checkout": Checkout(menu),
        }
    )
    agent = AgentSession[UserData](
        userdata=userdata,
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        # llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    await agent.start(
        agent=userdata.agents["greeter"],
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # await agent.say("Welcome to our restaurant! How may I assist you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
