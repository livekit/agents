import json
import os

from dotenv import load_dotenv

load_dotenv()

import asyncpg
from pydantic import BaseModel, Field
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
    inference,
)

RESTAURANT_SLUG = os.environ.get("RESTAURANT_SLUG", "bella-notte")


async def fetch_restaurant_and_menu(slug: str):
    """Loads a restaurant and its available menu items from Postgres."""
    conn = await asyncpg.connect(
        os.environ["DATABASE_URL"],
        statement_cache_size=0,
    )
    try:
        restaurant = await conn.fetchrow(
            "select id, name from restaurants where slug = $1", slug
        )
        if not restaurant:
            return None, []
        items = await conn.fetch(
            """
            select name, description, price, category
            from menu_items
            where restaurant_id = $1 and is_available = true
            order by category, name
            """,
            restaurant["id"],
        )
        return restaurant, items
    finally:
        await conn.close()


def format_menu_for_prompt(items) -> str:
    if not items:
        return "The menu is currently empty."
    by_category: dict[str, list[str]] = {}
    for item in items:
        line = f"{item['name']} (${item['price']:.2f})"
        if item["description"]:
            line += f" — {item['description']}"
        by_category.setdefault(item["category"], []).append(line)

    sections = []
    for category, lines in by_category.items():
        sections.append(f"{category}:\n" + "\n".join(f"  - {l}" for l in lines))
    return "\n\n".join(sections)


class OrderItem(BaseModel):
    name: str = Field(description="The exact menu item name, as listed on the menu.")
    quantity: int = Field(description="How many of this item the caller wants.", ge=1)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    restaurant, menu_items = await fetch_restaurant_and_menu(RESTAURANT_SLUG)

    if not restaurant:
        raise RuntimeError(
            f"No restaurant found with slug '{RESTAURANT_SLUG}'. "
            "Check RESTAURANT_SLUG and that it exists in the restaurants table."
        )

    menu_text = format_menu_for_prompt(menu_items)
    menu_by_name = {item["name"].lower(): item for item in menu_items}

    @function_tool
    async def get_menu(context: RunContext):
        """Returns the restaurant's current menu, grouped by category, with
        prices and descriptions. Use this whenever the caller asks what's
        available, what something costs, or for recommendations."""
        return {"restaurant": restaurant["name"], "menu": menu_text}

    @function_tool
    async def place_order(
        context: RunContext,
        items: list[OrderItem],
        customer_name: str | None = None,
        notes: str | None = None,
    ):
        """Places the caller's order. Only call this once you've read the
        full order back to the caller and they've confirmed it's correct.
        Every item name must match the menu exactly — if you're not sure,
        call get_menu first. `notes` is for special requests (e.g. "no
        onions", "extra spicy")."""
        order_lines = []
        unknown = []
        total = 0.0

        for entry in items:
            match = menu_by_name.get(entry.name.lower())
            if not match:
                unknown.append(entry.name)
                continue
            price = float(match["price"])
            order_lines.append(
                {"name": match["name"], "price": price, "quantity": entry.quantity}
            )
            total += price * entry.quantity

        if unknown:
            return {
                "error": (
                    f"These items aren't on the menu: {', '.join(unknown)}. "
                    "Let the caller know and offer real menu items instead."
                )
            }
        if not order_lines:
            return {"error": "No valid items were given — nothing was ordered."}

        conn = await asyncpg.connect(
            os.environ["DATABASE_URL"], statement_cache_size=0
        )
        try:
            await conn.execute(
                """
                insert into orders (restaurant_id, items, total, customer_name, notes)
                values ($1, $2::jsonb, $3, $4, $5)
                """,
                restaurant["id"],
                json.dumps(order_lines),
                total,
                customer_name,
                notes,
            )
        finally:
            await conn.close()

        return {
            "confirmation": f"Order placed at {restaurant['name']}.",
            "items": order_lines,
            "total": round(total, 2),
        }

    session = AgentSession(
        vad=inference.VAD(),
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS(
            "cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
    )

    agent = Agent(
        instructions=(
            f"You are the phone assistant for {restaurant['name']}, a "
            "restaurant. You help callers hear about menu items, prices, "
            "and take their order. Be warm, concise, and sound like a real "
            "host answering the phone — not like you're reading a list. "
            "Only talk about items that are actually on the menu below; "
            "if something isn't on it, say it's not available rather than "
            "guessing.\n\n"
            "When taking an order: confirm the full order (items, "
            "quantities, any special requests) back to the caller before "
            "calling place_order. After placing it, tell the caller the "
            "total and thank them.\n\n"
            f"Current menu:\n{menu_text}"
        ),
        tools=[get_menu, place_order],
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(
        instructions=f"Greet the caller as {restaurant['name']} and ask how you can help."
    )


if __name__ == "__main__":
    cli.run_app(server)