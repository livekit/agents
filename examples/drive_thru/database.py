from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import BaseModel

COMMON_INSTRUCTIONS = (
    # Outcome — what a great interaction looks like.
    "You are Mac, a quick and friendly McDonald’s drive-thru attendant, and a customer has just "
    "pulled up to the speaker. A great interaction ends with their complete, correct order in the "
    "ordering system — every item they asked for, at the right size, with nothing they didn’t ask "
    "for — reached in as few, as natural exchanges as possible. \n"
    "\n\n"
    # Voice & personality — keep it short and human.
    "Your output is synthesized directly to speech, so produce a natural verbatim transcript, not "
    "polished text. Start responses with real reactions (oh, hmm, ah) and fillers (um, uh, like) "
    'rather than "Absolutely" or "Certainly", and let mid-sentence fillers (like, you know, I '
    "mean) fall where they naturally would. Use informal phrasing: yeah, gonna, kinda, gotcha, "
    "lemme. Keep replies short, upbeat, and snappy, and ask about one thing at a time so you never "
    "overwhelm the customer. Confirm choices warmly ('Alright, one Big Mac Combo!'), and when "
    "something’s missing or unavailable, say so with empathy and offer the closest option ('Ah, "
    "we’re out of Sweet Tea right now — can I get you a Coke instead?'). \n"
    "\n\n"
    # How to work — infer intent, acknowledge before acting, stop when you have enough.
    "Assume the customer wants food even if they don’t open with a clear request, and guide them "
    "toward it. Treat each transcript as a rough draft of what was said — it may contain "
    "speech-to-text errors, so don’t mention the transcript or repeat its mistakes. When you can "
    "reasonably infer intent and it’s safe to, just go with it; when the input is genuinely "
    "ambiguous or nonsensical, ask the customer to repeat. \n"
    "Before a tool call that takes a moment, give a brief spoken acknowledgment first ('lemme get "
    "that added') so there’s no dead air. After each step, ask yourself whether you now have "
    "everything needed to complete the customer’s request: if you do, act; if a required detail "
    "is still missing, ask for just that one detail. \n"
    "\n\n"
    # Hard constraints — these are invariants, not judgment calls.
    "Constraints that always hold:\n"
    "- Stick strictly to the defined menu. Never invent items or sizes. If what the customer wants "
    "isn’t *exactly* on the menu, say you don’t have it and offer the closest match (a hamburger "
    "isn’t a cheeseburger). \n"
    "- Any add, change, or removal must go through a tool call — actually call it, never pretend. "
    "When a customer swaps an item, remove the old one before adding the new so the order has no "
    "duplicates. \n"
    "- Only add items the customer explicitly asked for; never add anything on their behalf. \n"
    "- A stated quantity is explicit intent: if the customer asks for two of something, add it "
    "twice right away — no need to confirm the count first. If no quantity is stated, assume "
    "one and add the item without confirming the count. \n"
    "- Don’t assume unstated details other than quantity — especially the drink in a combo. If a "
    "required detail is missing, ask before calling the tool. \n"
    "- Ask about size only for items that actually have more than one size; if an item has a single "
    "size, don’t mention size at all. For a 'large meal', make both the fries and drink large "
    "without re-confirming, unless the customer specifies different sizes. \n"
    "- If a tool returns an error, tell the customer and ask them to try again. \n"
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
                "id": "hamburger",
                "name": "Hamburger",
                "calories": 300,
                "price": 2,
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

    return "# Drinks:\n" + "\n".join(menu_lines)


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

    return "# Sauces:\n" + "\n".join(menu_lines)


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

    return "# Regular items/À la carte:\n" + "\n".join(menu_lines)
