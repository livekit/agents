from collections import defaultdict
from typing import Literal

from pydantic import BaseModel

COMMON_INSTRUCTIONS = (
    "You are Kelly, a quick and friendly McDonald’s drive-thru attendant. "
    "Your job is to guide the customer smoothly through their order, speaking in short, natural voice responses. "
    "This is a voice interaction-assume the customer just pulled up and is speaking to you through a drive-thru speaker. "
    "Respond like you're hearing them, not reading text. "
    "Assume they want food, even if they don’t start with a clear request, and help them get what they’re looking for. "
    "\n\n"
    "If an item comes in different sizes, always ask for the size unless the customer already gave one. "
    "If a customer orders a 'large meal', automatically assume both the fries and the drink should be large. "
    "Do not ask again to confirm the size of the drink or fries. This inference is meant to streamline the interaction. "
    "If the customer clearly indicates a different size for the fries or drink, respect their preference. "
    "\n\n"
    "Be fast-keep responses short and snappy. "
    "Sound human-sprinkle in light vocal pauses like 'Mmh…', 'Let me see…', or 'Alright…' at natural moments-but not too often. "
    "Keep everything upbeat and easy to follow. Never overwhelm the customer, don't ask multiple questions at the same time. "
    "\n\n"
    "When a customer is confused or asks for something that doesn’t exist, let them know politely and suggest something close. "
    "Always confirm what they picked in a warm, clear way, like: 'Alright, one Big Mac Combo!' "
    "If something’s unavailable, say so with empathy: 'Ah, we're out of Sweet Tea right now-can I get you a Coke instead?' "
    "\n\n"
    "Whenever a customer asks for, changes, or removes something from their order, you MUST use a tool to make it happen. "
    "Don’t fake it. Don’t pretend something was added - actually **call** the tool and make it real on the ordering system. "
    "\n\n"
    "Transcripts often contain speech-to-text errors-don’t mention the transcript, don’t repeat its mistakes. "
    "Instead treat each user input as a rough draft of what was said. "
    "If you can guess the user’s intent and it’s safe to do so, infer their meaning and respond naturally. "
    "If the transcript is ambiguous/nonsense and you can’t guess their intent, ask the customer to repeat again. "
    "Stay on-topic; if input is nonsensical in a drive-thru context, ask for concise clarification."
    "\n\n"
    "Do not add any item on the user's behalf unless they specifically request it. If the user hasn't asked for an item, NEVER add it."
    "\n\n"
    "When a customer changes an item or meal, make sure to remove the previous version before adding the new one. "
    "Otherwise, the order may contain duplicates."
    "\n\n"
    "Stricly stick to the defined menu, Do not invent or suggest any new sizes or items. "
    "Do not ask for size unless the item has more than one size option specified. "
    "If an item does not require a size according to the menu, **NEVER** ask the customer to choose one or mention size at all. "
)


ItemSize = Literal["S", "M", "L"]
ItemCategory = Literal["drink", "combo_meal", "happy_meal", "regular", "sauce"]


class MenuItem(BaseModel):
    id: str
    name: str
    calories: int
    price: float
    available: bool
    size: ItemSize | None = None
    voice_alias: str | None = None
    category: ItemCategory


class FakeDB:
    async def list_drinks(self) -> list[MenuItem]:
        drink_data = [
            {
                "id": "coca_cola",
                "name": "Coca-Cola®",
                "sizes": {
                    "S": {"calories": 200, "price": 1.49},
                    "M": {"calories": 270, "price": 1.69},
                    "L": {"calories": 380, "price": 1.89},
                },
            },
            {
                "id": "sprite",
                "name": "Sprite®",
                "sizes": {
                    "S": {"calories": 190, "price": 1.49},
                    "M": {"calories": 250, "price": 1.69},
                    "L": {"calories": 350, "price": 1.89},
                },
            },
            {
                "id": "diet_coke",
                "name": "Diet Coke®",
                "sizes": {
                    "S": {"calories": 0, "price": 1.49},
                    "M": {"calories": 0, "price": 1.69},
                    "L": {"calories": 0, "price": 1.89},
                },
            },
            {
                "id": "dr_pepper",
                "name": "Dr Pepper®",
                "sizes": {
                    "S": {"calories": 200, "price": 1.49},
                    "M": {"calories": 270, "price": 1.69},
                    "L": {"calories": 380, "price": 1.89},
                },
            },
            {
                "id": "fanta_orange",
                "name": "Fanta® Orange",
                "sizes": {
                    "S": {"calories": 210, "price": 1.49},
                    "M": {"calories": 280, "price": 1.69},
                    "L": {"calories": 390, "price": 1.89},
                },
            },
            {
                "id": "hi_c_orange_lavaburst",
                "name": "Hi-C® Orange Lavaburst®",
                "sizes": {
                    "S": {"calories": 210, "price": 1.49},
                    "M": {"calories": 280, "price": 1.69},
                    "L": {"calories": 390, "price": 1.89},
                },
            },
            {
                "id": "sweet_tea",
                "name": "Sweet Tea",
                "sizes": {
                    "S": {"calories": 140, "price": 1.39},
                    "M": {"calories": 180, "price": 1.59},
                    "L": {"calories": 220, "price": 1.79},
                },
                "available": False,
            },
            {
                "id": "unsweetened_iced_tea",
                "name": "Unsweetened Iced Tea",
                "sizes": {
                    "S": {"calories": 0, "price": 1.39},
                    "M": {"calories": 0, "price": 1.59},
                    "L": {"calories": 0, "price": 1.79},
                },
            },
            {
                "id": "minute_maid_orange_juice",
                "name": "Minute Maid® Premium Orange Juice",
                "sizes": {
                    "S": {"calories": 190, "price": 2.59},
                    "M": {"calories": 240, "price": 2.79},
                    "L": {"calories": 300, "price": 2.99},
                },
            },
            {
                "id": "milk",
                "name": "Milk",
                "calories": 100,
                "price": 1.29,
            },
            {
                "id": "chocolate_milk",
                "name": "Chocolate Milk",
                "calories": 150,
                "price": 1.39,
            },
            {
                "id": "dasani_water",
                "name": "DASANI® Water",
                "calories": 0,
                "price": 1.59,
            },
        ]

        items = []
        for item in drink_data:
            if sizes := item.get("sizes", {}):
                for size, size_details in sizes.items():
                    items.append(
                        MenuItem(
                            id=item["id"],
                            name=item["name"],
                            calories=size_details["calories"],
                            price=size_details["price"],
                            size=size,
                            available=True,
                            category="drink",
                        )
                    )
            else:
                items.append(
                    MenuItem(
                        id=item["id"],
                        name=item["name"],
                        calories=item["calories"],
                        price=item["price"],
                        available=True,
                        category="drink",
                    )
                )

        return items

    async def list_combo_meals(self) -> list[MenuItem]:
        raw_meals = [
            {
                "id": "combo_big_mac",
                "name": "Big Mac® Combo",
                "alias": "1",
                "calories": 970,
                "price": 9.49,
            },
            {
                "id": "combo_quarter_pounder_2a",
                "name": "Quarter Pounder® with Cheese Combo",
                "alias": "2a",
                "calories": 840,
                "price": 9.89,
            },
            {
                "id": "combo_quarter_pounder_2b",
                "name": "Quarter Pounder® with Cheese & Bacon Combo",
                "alias": "2b",
                "calories": 950,
                "price": 10.39,
            },
            {
                "id": "combo_quarter_pounder_2c",
                "name": "Quarter Pounder® Deluxe Combo",
                "alias": "2c",
                "calories": 950,
                "price": 10.39,
            },
            {
                "id": "combo_double_quarter",
                "name": "Double Quarter Pounder® with Cheese Combo",
                "alias": "3",
                "calories": 1060,
                "price": 10.29,
            },
            {
                "id": "combo_mccrispy_4a",
                "name": "McCrispy™ Original Combo",
                "alias": "4a",
                "calories": 790,
                "price": 8.99,
            },
            {
                "id": "combo_mccrispy_4b",
                "name": "McCrispy™ Spicy Combo",
                "alias": "4b",
                "calories": 850,
                "price": 8.99,
            },
            {
                "id": "combo_mccrispy_4c",
                "name": "McCrispy™ Deluxe Combo",
                "alias": "4c",
                "calories": 880,
                "price": 9.89,
            },
            {
                "id": "combo_mccrispy_4d",
                "name": "McCrispy™ Spicy Deluxe Combo",
                "alias": "4d",
                "calories": 860,
                "price": 9.99,
            },
            {
                "id": "combo_chicken_mcnuggets_10pc",
                "name": "10 pc. Chicken McNuggets® Combo",
                "alias": "5",
                "calories": 740,
                "price": 9.49,
            },
            {
                "id": "combo_filet_o_fish",
                "name": "Filet-O-Fish® Combo",
                "alias": "6",
                "calories": 700,
                "price": 7.89,
            },
            {
                "id": "combo_cheeseburgers_2pc",
                "name": "2 Cheeseburgers Combo",
                "alias": "7",
                "calories": 920,
                "price": 7.89,
            },
        ]

        meals = []

        for item in raw_meals:
            meals.append(
                MenuItem(
                    id=item["id"],
                    name=item["name"],
                    calories=item["calories"],
                    price=item["price"],
                    voice_alias=item["alias"],
                    category="combo_meal",
                    available=True,
                )
            )

        return meals

    async def list_happy_meals(self) -> list[MenuItem]:
        raw_happy_meals = [
            {
                "id": "happy_meal_4pc_mcnuggets",
                "name": "4 pc. Chicken McNuggets® Happy Meal",
                "calories": 430,
                "price": 5.99,
            },
            {
                "id": "happy_meal_6pc_mcnuggets",
                "name": "6 pc. Chicken McNuggets® Happy Meal",
                "calories": 530,
                "price": 6.99,
            },
            {
                "id": "happy_meal_hamburger",
                "name": "Hamburger Happy Meal",
                "calories": 510,
                "price": 5.59,
            },
        ]

        meals = []

        for item in raw_happy_meals:
            meals.append(
                MenuItem(
                    id=item["id"],
                    name=item["name"],
                    calories=item["calories"],
                    price=item["price"],
                    available=True,
                    category="happy_meal",
                )
            )

        return meals

    async def list_regulars(self) -> list[MenuItem]:
        raw_items = [
            {
                "id": "big_mac",
                "name": "Big Mac®",
                "calories": 590,
                "price": 5.89,
            },
            {
                "id": "quarter_pounder_cheese",
                "name": "Quarter Pounder® with Cheese",
                "calories": 520,
                "price": 6.29,
            },
            {
                "id": "quarter_pounder_bacon",
                "name": "Quarter Pounder® with Cheese & Bacon",
                "calories": 590,
                "price": 6.79,
            },
            {
                "id": "quarter_pounder_deluxe",
                "name": "Quarter Pounder® Deluxe",
                "calories": 530,
                "price": 6.39,
            },
            {
                "id": "double_quarter_pounder",
                "name": "Double Quarter Pounder® with Cheese",
                "calories": 740,
                "price": 7.49,
            },
            {
                "id": "mccrispy_original",
                "name": "McCrispy™ Original",
                "calories": 470,
                "price": 5.69,
            },
            {
                "id": "mccrispy_spicy",
                "name": "McCrispy™ Spicy",
                "calories": 500,
                "price": 5.69,
            },
            {
                "id": "mccrispy_deluxe",
                "name": "McCrispy™ Deluxe",
                "calories": 530,
                "price": 6.39,
            },
            {
                "id": "mccrispy_spicy_deluxe",
                "name": "McCrispy™ Spicy Deluxe",
                "calories": 530,
                "price": 6.59,
            },
            {
                "id": "mcnuggets_10pc",
                "name": "10 pc. Chicken McNuggets®",
                "calories": 410,
                "price": 6.79,
            },
            {
                "id": "filet_o_fish",
                "name": "Filet-O-Fish®",
                "calories": 390,
                "price": 5.89,
            },
            {
                "id": "cheeseburger",
                "name": "Cheeseburger",
                "calories": 600,
                "price": 2.58,
            },
            {
                "id": "fries",
                "name": "Fries",
                "sizes": {
                    "S": {"calories": 230, "price": 1.89},
                    "M": {"calories": 350, "price": 3.99},
                    "L": {"calories": 521, "price": 4.75},
                },
            },
            {
                "id": "sweet_sundae",
                "name": "Sundae",
                "calories": 330,
                "price": 3.69,
            },
            {
                "id": "sweet_mcflurry_oreo",
                "name": "McFlurry® (Oreo)",
                "calories": 480,
                "price": 4.89,
            },
            {
                "id": "shake_vanilla",
                "name": "Vanilla Shake",
                "sizes": {
                    "S": {"calories": 510, "price": 2.79},
                    "M": {"calories": 610, "price": 3.59},
                    "L": {"calories": 820, "price": 3.89},
                },
            },
            {
                "id": "shake_chocolate",
                "name": "Chocolate Shake",
                "sizes": {
                    "S": {"calories": 520, "price": 2.79},
                    "M": {"calories": 620, "price": 3.59},
                    "L": {"calories": 830, "price": 3.89},
                },
            },
            {
                "id": "shake_strawberry",
                "name": "Strawberry Shake",
                "sizes": {
                    "S": {"calories": 530, "price": 2.79},
                    "M": {"calories": 620, "price": 3.59},
                    "L": {"calories": 840, "price": 3.89},
                },
            },
            {
                "id": "sweet_cone",
                "name": "Cone",
                "calories": 200,
                "price": 3.19,
            },
        ]

        items = []
        for item in raw_items:
            if sizes := item.get("sizes", {}):
                for size, size_details in sizes.items():
                    items.append(
                        MenuItem(
                            id=item["id"],
                            name=item["name"],
                            calories=size_details["calories"],
                            price=size_details["price"],
                            size=size,
                            available=True,
                            category="regular",
                        )
                    )
            else:
                items.append(
                    MenuItem(
                        id=item["id"],
                        name=item["name"],
                        calories=item["calories"],
                        price=item["price"],
                        available=True,
                        category="regular",
                    )
                )

        return items

    async def list_sauces(self) -> list[MenuItem]:
        raw_items = [
            {
                "id": "jalapeno_ranch",
                "name": "Jalapeño Ranch",
                "calories": 70,
                "price": 0.25,
            },
            {
                "id": "garlic_sauce",
                "name": "Garlic Sauce",
                "calories": 45,
                "price": 0.25,
            },
            {
                "id": "mayonnaise",
                "name": "Mayonnaise",
                "calories": 90,
                "price": 0.20,
            },
            {
                "id": "frietsaus",
                "name": "Frietsaus",
                "calories": 100,
                "price": 0.20,
            },
            {
                "id": "curry_suace",
                "name": "Curry sauce",
                "calories": 60,
                "price": 0.20,
            },
            {
                "id": "ketchup",
                "name": "Ketchup",
                "calories": 20,
                "price": 0.10,
            },
            {
                "id": "barbecue_sauce",
                "name": "Barbecue Sauce",
                "calories": 45,
                "price": 0.20,
            },
            {
                "id": "sweet_and_sour_sauce",
                "name": "Sweet-and-sour sauce",
                "calories": 50,
                "price": 0.40,
            },
            {
                "id": "honey_mustard_dressing",
                "name": "Honey mustard dressing",
                "calories": 60,
                "price": 0.20,
            },
        ]
        sauces = []

        for item in raw_items:
            sauces.append(
                MenuItem(
                    id=item["id"],
                    name=item["name"],
                    calories=item["calories"],
                    price=item["price"],
                    available=True,
                    category="sauce",
                )
            )

        return sauces


# The code below is optimized for ease of use instead of efficiency.


def map_by_sizes(
    items: list[MenuItem],
) -> tuple[dict[str, dict[ItemSize, MenuItem]], list[MenuItem]]:
    result = defaultdict(dict)
    leftovers = [item for item in items if not item.size]
    [result[item.id].update({item.size: item}) for item in items if item.size]
    return dict(result), leftovers


def find_items_by_id(
    items: list[MenuItem], item_id: str, size: ItemSize | None = None
) -> list[MenuItem]:
    return [item for item in items if item.id == item_id and (size is None or item.size == size)]


def menu_instructions(category: ItemCategory, *, items: list[MenuItem]) -> str:
    if category == "drink":
        return _drink_menu_instructions(items)
    elif category == "combo_meal":
        return _combo_menu_instructions(items)
    elif category == "happy_meal":
        return _happy_menu_instructions(items)
    elif category == "sauce":
        return _sauce_menu_instructions(items)
    elif category == "regular":
        return _regular_menu_instructions(items)


def _drink_menu_instructions(items: list[MenuItem]) -> str:
    available_sizes, leftovers = map_by_sizes(items)
    menu_lines = []

    for _, size_map in available_sizes.items():
        first_item = next(iter(size_map.values()))
        menu_lines.append(f"  - {first_item.name} (id:{first_item.id}):")

        for item in size_map.values():
            line = f"    - Size {item.size}: {item.calories} Cal, ${item.price:.2f}"
            if not item.available:
                line += " UNAVAILABLE"
            menu_lines.append(line)

    for item in leftovers:
        # explicitely saying there is no `size` for this item, otherwise the LLM seems to hallucinate quite often
        line = f"  - {item.name}: {item.calories} Cal, ${item.price:.2f} (id:{item.id}) - Not size-selectable`"
        if not item.available:
            line += " UNAVAILABLE"
        menu_lines.append(line)

    return f"# Drinks:\n{'\n'.join(menu_lines)}"


def _combo_menu_instructions(items: list[MenuItem]) -> str:
    menu_lines = []
    for item in items:
        line = f"  **{item.voice_alias}**. {item.name}: {item.calories} Cal, ${item.price:.2f} (id:{item.id})"

        if not item.available:
            line += " UNAVAILABLE"
        menu_lines.append(line)

    instructions = (
        "# Combo Meals:\n"
        "The user can select a combo meal by saying its voice alias (e.g., '1', '2a', '4c'). Use the alias to identify which combo they chose.\n"
        "But don't mention the voice alias to the user if not needed."
    )
    return instructions + "\n".join(menu_lines)


def _happy_menu_instructions(items: list[MenuItem]) -> str:
    menu_lines = []
    for item in items:
        line = f"  - {item.name}: {item.calories} Cal, ${item.price:.2f} (id:{item.id})"
        if not item.available:
            line += " UNAVAILABLE"
        menu_lines.append(line)

    return (
        "# Happy Meals:\n" + "\n".join(menu_lines) + "\n\nRecommended drinks with the Happy Meal:\n"
        "  - Milk chocolate/white\n"
        "  - DASANI Water\n"
        "  - Or any other small drink."
    )


def _sauce_menu_instructions(items: list[MenuItem]) -> str:
    menu_lines = []
    for item in items:
        line = f"  - {item.name}: {item.calories} Cal, ${item.price:.2f} (id:{item.id})"
        if not item.available:
            line += " UNAVAILABLE"
        menu_lines.append(line)

    return f"# Sauces:\n{'\n'.join(menu_lines)}"


# regular/a la carte
def _regular_menu_instructions(items: list[MenuItem]) -> str:
    available_sizes, leftovers = map_by_sizes(items)
    menu_lines = []

    for _, size_map in available_sizes.items():
        first_item = next(iter(size_map.values()))
        menu_lines.append(f"  - {first_item.name} (id:{first_item.id}):")

        for item in size_map.values():
            line = f"    - Size {item.size}: {item.calories} Cal, ${item.price:.2f}"
            if not item.available:
                line += " UNAVAILABLE"
            menu_lines.append(line)

    for item in leftovers:
        line = f"  - {item.name}: {item.calories} Cal, ${item.price:.2f} (id:{item.id}) - Not size-selectable"
        if not item.available:
            line += " UNAVAILABLE"
        menu_lines.append(line)

    return f"# Regular items/À la carte:\n{'\n'.join(menu_lines)}"
