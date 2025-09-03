import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Union

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    ToolError,
    WorkerOptions,
    cli,
    function_tool,
    llm,
)
from livekit.agents.voice.events import (
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
)
from livekit.plugins import silero
from livekit.plugins.ultravox.realtime import RealtimeModel

logger = logging.getLogger("ultravox-agent")
logger.setLevel(logging.INFO)

load_dotenv()


@function_tool
async def get_time(
    time_zone: Annotated[str, "The time zone to get the time in"],
) -> str:
    """
    Get the current time in the given time zone.

    Args:
        time_zone: The time zone to get the time in.

    Returns:
        The current time in the given time zone.
    """
    try:
        result = datetime.now(timezone.utc).isoformat()
        logger.info(f"[tool:get_time] tz={time_zone!r} -> {result}")
        return result
    except Exception as e:
        logger.error(f"[tool:get_time] EXCEPTION: {e}", exc_info=True)
        return f"Error getting time: {e}"


@function_tool(
    raw_schema={
        "name": "get_time_raw",
        "description": "Get the current time in the given time zone.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_zone": {"type": "string", "description": "The time zone to get the time in"},
            },
            "required": ["time_zone"],
        },
    }
)
async def get_time_raw(
    raw_arguments: dict[str, object],
) -> str:
    """
    Get the current time in the given time zone.

    Args:
        time_zone: The time zone to get the time in.

    Returns:
        The current time in the given time zone.
    """
    try:
        result = datetime.now(timezone.utc).isoformat()
        logger.info(f"[tool:get_time_raw] args={raw_arguments} -> {result}")
        return result
    except Exception as e:
        logger.error(f"[tool:get_time_raw] EXCEPTION: {e}", exc_info=True)
        return f"Error getting time: {e}"


@function_tool
async def say_hello(name: str) -> str:
    """Greets the user by name."""
    try:
        result = f"Hello, {name}!"
        logger.info(f"[tool:say_hello] name={name!r} -> {result}")
        return result
    except Exception as e:
        logger.error(f"[tool:say_hello] EXCEPTION: {e}", exc_info=True)
        return f"Error saying hello: {e}"


class PizzaSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


@function_tool
async def order_pizza(
    size: Annotated[PizzaSize, "The size of the pizza"],
    toppings: Annotated[Union[list[str], str], "A list of toppings (array or comma-separated)"],
    quantity: Annotated[int, "How many pizzas to order"] = 1,
) -> str:
    """Orders a pizza with the specified size and toppings.

    The `toppings` argument may be provided either as a JSON array (preferred) or
    as a comma-separated string. This lenient parsing avoids validation errors
    when the LLM accidentally sends a string representation like "[... ]".
    """
    try:
        logger.info(
            f"[tool:order_pizza] ENTRY: size={size}, toppings={toppings}, quantity={quantity}"
        )

        # Normalise toppings into a list[str]
        if isinstance(toppings, str):
            import json
            import re

            # Try to parse JSON first
            try:
                parsed = json.loads(toppings)
                if isinstance(parsed, list):
                    toppings = [str(t).strip() for t in parsed]
                else:
                    # Fallback: split by comma / spaces
                    toppings = [s.strip() for s in re.split(r",|\s+", toppings) if s.strip()]
            except json.JSONDecodeError:
                toppings = [s.strip() for s in toppings.split(",") if s.strip()]

        toppings_str = ", ".join(toppings) if toppings else "no toppings"
        result = (
            f"Ordered {quantity} {size.value} pizza(s) with {toppings_str}. "
            "Your order will arrive shortly!"
        )
        logger.info(f"[tool:order_pizza] SUCCESS -> {result}")
        return result
    except Exception as e:
        logger.error(f"[tool:order_pizza] EXCEPTION: {e}", exc_info=True)
        return f"Error ordering pizza: {e}"


@function_tool
async def check_availability(item: str) -> str:
    """Checks store inventory and responds to the user.

    The tool always returns a string for the LLM to relay, never raises an
    exception. This keeps the interaction smooth and avoids generic apology
    fallbacks from the framework.
    """
    try:
        logger.info(f"[tool:check_availability] ENTRY: item={item!r}")

        if "unicorn" in item.lower():
            result = "Sorry, we are all out of unicorns at the moment."
        else:
            result = f"Yes, {item} is available!"

        logger.info(f"[tool:check_availability] SUCCESS -> {result}")
        return result
    except Exception as e:
        logger.error(f"[tool:check_availability] EXCEPTION: {e}", exc_info=True)
        return f"Error checking availability: {e}"


# Simple knowledge base for testing
CUSTOMER_DATABASE = {
    "john smith": {
        "name": "John Smith",
        "account_id": "ACC-001",
        "status": "VIP",
        "balance": "$1,250",
    },
    "jane doe": {
        "name": "Jane Doe",
        "account_id": "ACC-002",
        "status": "Standard",
        "balance": "$340",
    },
}

KNOWLEDGE_BASE = {
    "hours": "Our business hours are Monday-Friday 9AM-6PM EST, Saturday 10AM-4PM EST.",
    "pricing": "Standard support: $50/month, Premium: $150/month, Enterprise: custom pricing",
    "support": "For urgent issues, call 1-800-HELP. For billing, email billing@company.com",
}


@function_tool
async def lookup_customer(
    customer_name: Annotated[str, "The customer's full name to look up"],
) -> str:
    """Look up customer information by name."""
    try:
        customer_data = CUSTOMER_DATABASE.get(customer_name.lower())
        if customer_data:
            result = f"Customer: {customer_data['name']}, Account: {customer_data['account_id']}, Status: {customer_data['status']}, Balance: {customer_data['balance']}"
            logger.info(f"[tool:lookup_customer] Found: {customer_data['name']}")
            return result
        else:
            return f"No customer found with name '{customer_name}'"
    except Exception as e:
        logger.error(f"[tool:lookup_customer] Error: {e}")
        raise ToolError(f"Error looking up customer: {e}") from e


@function_tool
async def search_knowledge(
    query: Annotated[str, "The topic to search for (hours, pricing, support, etc.)"],
) -> str:
    """Search the company knowledge base for information."""
    try:
        query_lower = query.lower()
        for key, value in KNOWLEDGE_BASE.items():
            if key in query_lower:
                logger.info(f"[tool:search_knowledge] Found info for: {key}")
                return value
        return f"No information found for '{query}'. Available topics: hours, pricing, support"
    except Exception as e:
        logger.error(f"[tool:search_knowledge] Error: {e}")
        raise ToolError(f"Error searching knowledge base: {e}") from e


class TestAgent(Agent):
    """Test agent that demonstrates both tools and dynamic context management."""

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: llm.ChatMessage
    ) -> None:
        """Inject relevant context based on user input - demonstrates RAG integration."""
        try:
            user_text = new_message.content.lower() if new_message.content else ""
            logger.info(f"[RAG] Processing user turn: {user_text[:50]}...")

            # Auto-inject customer context if name mentioned
            for name in CUSTOMER_DATABASE:
                if name in user_text:
                    customer_data = CUSTOMER_DATABASE[name]
                    logger.info(f"[RAG] Auto-injecting context for: {customer_data['name']}")
                    turn_ctx.add_message(
                        role="system",
                        content=f"Context: Customer {customer_data['name']} is calling. Account: {customer_data['account_id']}, Status: {customer_data['status']}, Balance: {customer_data['balance']}",
                    )
                    break

            # Auto-inject knowledge if relevant topic mentioned
            for topic in KNOWLEDGE_BASE:
                if topic in user_text:
                    logger.info(f"[RAG] Auto-injecting knowledge for: {topic}")
                    turn_ctx.add_message(role="system", content=f"Context: {KNOWLEDGE_BASE[topic]}")
                    break

        except Exception as e:
            logger.error(f"[RAG] Error in context injection: {e}", exc_info=True)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session: AgentSession[None] = AgentSession(
        allow_interruptions=True,
        vad=ctx.proc.userdata["vad"],
        llm=RealtimeModel(
            model_id="fixie-ai/ultravox",
            voice="Jessica",
            temperature=0.7,
            language_hint="en",
        ),
    )

    @session.on("function_tools_executed")
    def on_function_tools_executed(ev: FunctionToolsExecutedEvent) -> None:
        logger.info(f"function_tools_executed: {ev}")

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        logger.info(f"conversation_item_added: {ev}")

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        # Use framework helper for consistent formatting
        logger.info(f"metrics_collected: {ev}")

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        logger.info(f"user_input_transcribed: {ev}")

    await session.start(
        agent=TestAgent(
            instructions="""You are a helpful customer service assistant.

You have access to:
1. Customer lookup tools (use when customers give their name)
2. Knowledge base search (for company info like hours, pricing, support)
3. Basic tools like time and availability checking

When you receive context about a customer or company information, acknowledge it and use it in your response.
Be conversational and helpful. Use tools when needed to look up specific information.""",
            tools=[
                get_time,
                get_time_raw,
                say_hello,
                order_pizza,
                check_availability,
                lookup_customer,
                search_knowledge,
            ],
        ),
        room=ctx.room,
    )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
