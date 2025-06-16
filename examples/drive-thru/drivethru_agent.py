import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    function_tool,
    JobContext,
    cli,
    WorkerOptions,
    AgentSession,
    RunContext,
    ToolError,
    FunctionTool,
    AudioConfig,
    BackgroundAudioPlayer,
)

from typing import Literal, Annotated, Union
from dataclasses import dataclass
from database import (
    COMMON_INSTRUCTIONS,
    find_items_by_id,
    menu_instructions,
    FakeDB,
    MenuItem,
)
from pydantic import BaseModel, Field
from order import OrderState, OrderedCombo, OrderedHappy, OrderedRegular
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from livekit.plugins import deepgram, openai, elevenlabs, silero, cartesia


load_dotenv()


@dataclass
class Userdata:
    order: OrderState
    drink_items: list[MenuItem]
    combo_items: list[MenuItem]
    happy_items: list[MenuItem]
    regular_items: list[MenuItem]
    sauce_items: list[MenuItem]


class DriveThruAgent(Agent):
    def __init__(self, *, userdata: Userdata) -> None:
        instructions = (
            COMMON_INSTRUCTIONS
            + "\n\n"
            + menu_instructions("drink", items=userdata.drink_items)
            + "\n\n"
            + menu_instructions("combo_meal", items=userdata.combo_items)
            + "\n\n"
            + menu_instructions("happy_meal", items=userdata.happy_items)
            + "\n\n"
            + menu_instructions("regular", items=userdata.regular_items)
            + "\n\n"
            + menu_instructions("sauce", items=userdata.sauce_items)
        )

        super().__init__(
            instructions=instructions,
            tools=[
                self.build_regular_order_tool(
                    userdata.regular_items, userdata.drink_items, userdata.sauce_items
                ),
                self.build_combo_order_tool(
                    userdata.combo_items, userdata.drink_items, userdata.sauce_items
                ),
                self.build_happy_order_tool(
                    userdata.happy_items, userdata.drink_items, userdata.sauce_items
                ),
            ],
        )

    def build_combo_order_tool(
        self, combo_items: list[MenuItem], drink_items: list[MenuItem], sauce_items: list[MenuItem]
    ) -> FunctionTool:
        available_combo_ids = {item.id for item in combo_items}
        available_drink_ids = {item.id for item in drink_items}
        available_sauce_ids = {item.id for item in sauce_items}

        @function_tool
        async def order_combo_meal(
            ctx: RunContext[Userdata],
            meal_id: Annotated[
                str,
                Field(
                    description="The ID of the combo meal the user requested.",
                    json_schema_extra={"enum": list(available_combo_ids)},
                ),
            ],
            drink_id: Annotated[
                str,
                Field(
                    description="The ID of the drink the user requested.",
                    json_schema_extra={"enum": list(available_drink_ids)},
                ),
            ],
            drink_size: Literal["M", "L", "null"] | None,
            fries_size: Literal["M", "L"],
            sauce_id: Annotated[
                str,
                Field(
                    description="The ID of the sauce the user requested.",
                    json_schema_extra={"enum": list(available_sauce_ids)},
                ),
            ],
        ):
            """
            Call this when the user orders a **Combo Meal**, like: “Number 4b with a large Sprite” or “I'll do a medium meal.”

            Do not call this tool unless the user clearly refers to a known combo meal by name or number.
            Regular items like a single cheeseburger cannot be made into a meal unless such a combo explicitly exists.

            Only call this function once the user has clearly specified both a drink and a sauce — always ask for them if they're missing.

            A meal can only be Medium or Large; Small is not an available option.
            Drink and fries sizes can differ (e.g., “large fries but a medium Coke”).

            If the user says just “a large meal,” assume both drink and fries are that size.
            """
            if not find_items_by_id(combo_items, meal_id):
                raise ToolError(f"error: the meal {meal_id} was not found")

            drink_sizes = find_items_by_id(drink_items, drink_id)
            if not drink_sizes:
                raise ToolError(f"error: the drink {drink_id} was not found")

            if drink_size == "null":
                drink_size = None

            available_sizes = list({item.size for item in drink_sizes if item.size})
            if drink_size is None and len(available_sizes) > 1:
                raise ToolError(
                    f"error: {drink_id} comes with multiple sizes: {', '.join(available_sizes)}. "
                    "Please clarify which size should be selected."
                )

            if drink_size is not None and not available_sizes:
                raise ToolError(
                    f"error: size should not be specified for item {drink_id} as it does not support sizing options."
                )

            available_sizes = list({item.size for item in drink_sizes if item.size})
            if drink_size not in available_sizes:
                drink_size = None
                # raise ToolError(
                #     f"error: unknown size {drink_size} for {drink_id}. Available sizes: {', '.join(available_sizes)}."
                # )

            if not find_items_by_id(sauce_items, sauce_id):
                raise ToolError(f"error: the sauce {sauce_id} was not found")

            item = OrderedCombo(
                meal_id=meal_id,
                drink_id=drink_id,
                drink_size=drink_size,
                sauce_id=sauce_id,
                fries_size=fries_size,
            )
            await ctx.userdata.order.add(item)
            return f"The item was added: {item.model_dump_json()}"

        return order_combo_meal

    def build_happy_order_tool(
        self,
        happy_items: list[MenuItem],
        drink_items: list[MenuItem],
        sauce_items: list[MenuItem],
    ) -> FunctionTool:
        available_happy_ids = {item.id for item in happy_items}
        available_drink_ids = {item.id for item in drink_items}
        available_sauce_ids = {item.id for item in sauce_items}

        @function_tool
        async def order_happy_meal(
            ctx: RunContext[Userdata],
            meal_id: Annotated[
                str,
                Field(
                    description="The ID of the happy meal the user requested.",
                    json_schema_extra={"enum": list(available_happy_ids)},
                ),
            ],
            drink_id: Annotated[
                str,
                Field(
                    description="The ID of the drink the user requested.",
                    json_schema_extra={"enum": list(available_drink_ids)},
                ),
            ],
            drink_size: Literal["S", "M", "L", "null"] | None,
            sauce_id: Annotated[
                str,
                Field(
                    description="The ID of the sauce the user requested.",
                    json_schema_extra={"enum": list(available_sauce_ids)},
                ),
            ],
        ) -> str:
            """
            Call this when the user orders a **Happy Meal**, typically for children. These meals come with a main item, a drink, and a sauce.

            The user must clearly specify a valid Happy Meal option (e.g., “Can I get a Happy Meal?”).

            Before calling this tool:
            - Ensure the user has provided all required components: a valid meal, drink, drink size, and sauce.
            - If any of these are missing, prompt the user for the missing part before proceeding.

            Assume Small as default only if the user says "Happy Meal" and gives no size preference, but always ask for clarification if unsure.
            """
            if not find_items_by_id(happy_items, meal_id):
                raise ToolError(f"error: the meal {meal_id} was not found")

            drink_sizes = find_items_by_id(drink_items, drink_id)
            if not drink_sizes:
                raise ToolError(f"error: the drink {drink_id} was not found")

            if drink_size == "null":
                drink_size = None

            available_sizes = list({item.size for item in drink_sizes if item.size})
            if drink_size is None and len(available_sizes) > 1:
                raise ToolError(
                    f"error: {drink_id} comes with multiple sizes: {', '.join(available_sizes)}. "
                    "Please clarify which size should be selected."
                )

            if drink_size is not None and not available_sizes:
                drink_size = None

            if not find_items_by_id(sauce_items, sauce_id):
                raise ToolError(f"error: the sauce {sauce_id} was not found")

            item = OrderedHappy(
                meal_id=meal_id,
                drink_id=drink_id,
                drink_size=drink_size,
                sauce_id=sauce_id,
            )
            await ctx.userdata.order.add(item)
            return f"The item was added: {item.model_dump_json()}"

        return order_happy_meal

    def build_regular_order_tool(
        self,
        regular_items: list[MenuItem],
        drink_items: list[MenuItem],
        sauce_items: list[MenuItem],
    ) -> FunctionTool:
        all_items = regular_items + drink_items + sauce_items
        available_ids = {item.id for item in all_items}

        @function_tool
        async def order_regular_item(
            ctx: RunContext[Userdata],
            item_id: Annotated[
                str,
                Field(
                    description="The ID of the item the user requested.",
                    json_schema_extra={"enum": list(available_ids)},
                ),
            ],
            size: Annotated[
                # models don't seem to understand `ItemSize | None`, adding the `null` inside the enum list as a workaround
                Literal["S", "M", "L", "null"] | None,
                Field(
                    description="Size of the item, if applicable (e.g., 'S', 'M', 'L'), otherwise 'null'. "
                ),
            ] = "null",
        ) -> str:
            """
            Call this when the user orders **a single item on its own**, not as part of a Combo Meal or Happy Meal.

            The customer must provide clear and specific input. For example, item variants such as flavor must **always** be explicitly stated.

            The user might say—for example:
            - “Just the cheeseburger, no meal”
            - “A medium Coke”
            - “Can I get some ketchup?”
            - “Can I get a McFlurry Oreo?”
            """
            item_sizes = find_items_by_id(all_items, item_id)
            if not item_sizes:
                raise ToolError(f"error: {item_id} was not found.")

            if size == "null":
                size = None

            available_sizes = list({item.size for item in item_sizes if item.size})
            if size is None and len(available_sizes) > 1:
                raise ToolError(
                    f"error: {item_id} comes with multiple sizes: {', '.join(available_sizes)}. "
                    "Please clarify which size should be selected."
                )

            if size is not None and not available_sizes:
                size = None
                # raise ToolError(
                #     f"error: size should not be specified for item {item_id} as it does not support sizing options."
                # )

            if (size and available_sizes) and size not in available_sizes:
                raise ToolError(
                    f"error: unknown size {size} for {item_id}. Available sizes: {', '.join(available_sizes)}."
                )

            item = OrderedRegular(item_id=item_id, size=size)
            await ctx.userdata.order.add(item)
            return f"The item was added: {item.model_dump_json()}"

        return order_regular_item

    @function_tool
    async def remove_order_item(
        self,
        ctx: RunContext[Userdata],
        order_id: Annotated[
            list[str],
            Field(
                description="A list of internal `order_id`s of the items to remove. Use `list_order_items` to look it up if needed."
            ),
        ],
    ) -> str:
        """
        Removes one or more items from the user's order using their `order_id`s.

        Useful when the user asks to cancel or delete existing items (e.g., “Remove the cheeseburger”).

        If the `order_id`s are unknown, call `list_order_items` first to retrieve them.
        """
        not_found = [oid for oid in order_id if oid not in ctx.userdata.order.items]
        if not_found:
            raise ToolError(f"error: no item(s) found with order_id(s): {', '.join(not_found)}")

        removed_items = [await ctx.userdata.order.remove(oid) for oid in order_id]
        return "Removed items:\n" + "\n".join(item.model_dump_json() for item in removed_items)

    @function_tool
    async def list_order_items(self, ctx: RunContext[Userdata]) -> str:
        """
        Retrieves the current list of items in the user's order, including each item's internal `order_id`.

        Helpful when:
        - An `order_id` is required before modifying or removing an existing item.
        - Confirming details or contents of the current order.

        Examples:
        - User requests modifying an item, but the item's `order_id` is unknown (e.g., "Change the fries from small to large").
        - User requests removing an item, but the item's `order_id` is unknown (e.g., "Remove the cheeseburger").
        - User asks about current order details (e.g., "What's in my order so far?").
        """
        items = ctx.userdata.order.items.values()
        if not items:
            return "The order is empty"

        return "\n".join(item.model_dump_json() for item in items)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    fake_db = FakeDB()
    drink_items = await fake_db.list_drinks()
    combo_items = await fake_db.list_combo_meals()
    happy_items = await fake_db.list_happy_meals()
    regular_items = await fake_db.list_regulars()
    sauce_items = await fake_db.list_sauces()

    order_state = OrderState(items={})
    userdata = Userdata(
        order=order_state,
        drink_items=drink_items,
        combo_items=combo_items,
        happy_items=happy_items,
        regular_items=regular_items,
        sauce_items=sauce_items,
    )
    session = AgentSession[Userdata](
        userdata=userdata,
        stt=deepgram.STT(
            model="nova-3",
            keyterms=[
                "Big Mac",
                "McFlurry",
                "McCrispy",
                "McNuggets",
                "Meal",
                "Sundae",
                "Oreo",
                "Jalapeno Ranch",
            ],
            mip_opt_out=True,
        ),
        llm=openai.LLM(model="gpt-4o"),
        # tts=elevenlabs.TTS(
        #     model="eleven_turbo_v2_5",
        #     voice_id="21m00Tcm4TlvDq8ikWAM",
        #     voice_settings=elevenlabs.VoiceSettings(
        #         speed=1.15, stability=0.5, similarity_boost=0.75
        #     ),
        # ),
        tts=cartesia.TTS(voice="f786b574-daa5-4673-aa0c-cbe3e8534c02", speed="fast"),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        max_tool_steps=10,
    )

    background_audio = BackgroundAudioPlayer(
        ambient_sound=AudioConfig(
            str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bg_noise.mp3")),
            volume=1.0,
        ),
    )

    await session.start(agent=DriveThruAgent(userdata=userdata), room=ctx.room)
    await background_audio.start(room=ctx.room, agent_session=session)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
