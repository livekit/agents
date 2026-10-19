from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from agent import (
    DeliveryUpdateAgent,
    _answered_yes_since,
    _asked_for_more,
    _cancel_needs_a_yes,
    _cancel_read_back_at,
    _drink_size_said,
    _drop_off_unsaved,
    _line_note,
    _money,
    _opener,
    _read_back_answer,
    _voicemail,
    say_pepperoni_pizzas,
)
from backend import Pos, PosError, demo_pos, looks_like_an_address
from call import (
    SILENT_GOODBYE,
    STILL_THERE,
    hang_up_when_silent,
    spoken_number,
)
from grade import grade
from page import Page

from livekit.agents import llm
from livekit.agents.voice import UserStateChangedEvent

pytestmark = pytest.mark.unit


@pytest.fixture()
def pos() -> Pos:
    return demo_pos()


def _ready_for_pickup(pos: Pos) -> None:
    pos.set_pickup()
    pos.set_contact("sam carter", "415-555-0142")


def test_menu_is_the_seeded_day(pos: Pos) -> None:
    assert len(pos.menu) == 31
    assert not pos.item("buffalo_cauliflower").available
    assert "nuts" in pos.item("pesto").allergens


def test_ro01_pickup_order_matches_and_totals(pos: Pos) -> None:
    pos.add("pepperoni", 1, "large")
    pos.add("wings", 1, "a dozen", sauce="hot")
    pos.add("coke", 1, "large")
    _ready_for_pickup(pos)
    q = pos.price()
    # 19.99 + 16.99 + 3.99, then 9% tax rounded to the cent
    assert q.subtotal == Decimal("40.97")
    assert q.tax == Decimal("3.69")
    assert q.delivery_fee == 0
    assert q.total == Decimal("44.66")
    order = pos.submit()
    assert order.number
    assert [ln.describe() for ln in pos.lines] == [
        "L1: 1 × large Pepperoni",
        "L2: 1 × dozen Bone-in Wings, hot sauce",
        "L3: 1 × large Coke",
    ]


def test_ro02_corrections_mid_read_back_change_only_those_lines(pos: Pos) -> None:
    pos.add("pepperoni", 1, "L")
    pos.add("wings", 1, "12", sauce="thai")
    pos.add("coke", 1, "large")
    _ready_for_pickup(pos)
    pos.price()
    pos.update("L1", size="medium")
    pos.remove("L3")
    # the read-back total is stale now, so it can't be submitted at the old price
    assert not pos.quote_current
    with pytest.raises(PosError, match="changed after it was read back"):
        pos.submit()
    q = pos.price()
    assert q.subtotal == Decimal("16.99") + Decimal("16.99")
    pos.submit()
    assert [ln.describe() for ln in pos.lines] == [
        "L1: 1 × medium Pepperoni",
        "L2: 1 × dozen Bone-in Wings, Thai chili peanut sauce",
    ]
    assert [ln.id for ln in pos.removed] == ["L3"]
    assert pos.lines[0].was == "1 × large Pepperoni"
    assert pos.lines[0].rev == 2


def test_ro03_86d_item_is_refused_with_same_category_substitutes(pos: Pos) -> None:
    with pytest.raises(PosError, match="sold out today") as e:
        pos.add("buffalo_cauliflower")
    assert "Breadsticks" in str(e.value) and "Pepperoni" not in str(e.value)
    assert "86" not in str(e.value)  # kitchen slang, not for the caller
    assert not pos.lines


def test_ro04_delivers_to_any_street_address(pos: Pos) -> None:
    pos.add("pepperoni", 2, "large")
    pos.set_delivery("2150 Bryant St, San Francisco 94110")
    assert (pos.fulfillment.mode, pos.fulfillment.zip) == ("delivery", "94110")
    pos.set_delivery("45 Rideau St, Ottawa, ON K1N 5W8")
    assert pos.fulfillment.address == "45 Rideau St, Ottawa, ON K1N 5W8"
    assert pos.fulfillment.zip is None
    # the driver needs a street: a ZIP, a city, or a street with no number isn't one
    for partial in ("94107", "Oakland", "De Haro St"):
        with pytest.raises(PosError, match="full street address"):
            pos.set_delivery(partial)
    assert pos.fulfillment.address == "45 Rideau St, Ottawa, ON K1N 5W8"
    pos.set_pickup()
    assert pos.fulfillment.mode == "pickup"


def test_ro05_a_small_delivery_order_goes_in_with_the_small_order_fee(pos: Pos) -> None:
    # no delivery minimum: a medium cheese ($14.99) delivers, with the $2.99 fee on top
    pos.add("cheese", 1, "medium")
    pos.set_delivery("300 De Haro St, San Francisco, CA 94107")
    pos.set_contact("Sam", "(415) 555-0142")
    q = pos.price()
    assert (q.subtotal, q.tax, q.delivery_fee, q.small_order_fee) == (
        Decimal("14.99"),
        Decimal("1.35"),
        Decimal("4.99"),
        Decimal("2.99"),
    )
    assert q.total == Decimal("24.32")
    assert "small-order fee $2.99" in q.describe()
    assert pos.snapshot()["quote"]["smallOrderFee"] == 2.99
    assert "shortOfMinimum" not in pos.snapshot()["fulfillment"]
    pos.submit()
    assert grade(pos, {"submitted": True, "total": "24.32"}) == []


def test_the_small_order_fee_is_for_pickup_too_and_only_under_15(pos: Pos) -> None:
    pos.add("fries")  # $4.49
    pos.set_pickup()
    q = pos.price()
    assert q.small_order_fee == Decimal("2.99")
    # untaxed, like the delivery fee
    assert q.total == Decimal("4.49") + Decimal("0.40") + Decimal("2.99")
    pos.add("house_salad")  # $12.48: still under
    assert pos.price().small_order_fee == Decimal("2.99")
    pos.add("cannoli")  # $16.97
    q = pos.price()
    assert q.small_order_fee == 0 and "small-order" not in q.describe()
    assert pos.snapshot()["quote"]["smallOrderFee"] == 0


def test_ro06_allergen_lookup_names_the_nut_items(pos: Pos) -> None:
    tag, items = pos.with_allergen("peanuts")
    assert tag == "nuts"
    assert items == ["Pesto Chicken", "wings with Thai chili peanut sauce"]
    with pytest.raises(PosError, match="doesn't track"):
        pos.with_allergen("sesame")
    line = pos.add("wings", 1, "6", sauce="Thai")
    assert "nuts" in line.allergens


@pytest.mark.parametrize(
    ("said", "tag"),
    [
        ("gluten-free", "gluten"),
        ("Gluten free", "gluten"),
        ("celiac", "gluten"),
        ("I have celiac disease", "gluten"),
        ("dairy-free", "dairy"),
        ("lactose intolerant", "dairy"),
        ("peanut allergy", "nuts"),
        ("tree nuts", "nuts"),
    ],
)
def test_allergy_phrasings_map_to_the_tracked_allergens(pos: Pos, said: str, tag: str) -> None:
    assert pos.with_allergen(said)[0] == tag


def test_untracked_or_mixed_allergens_are_refused(pos: Pos) -> None:
    for said in ("sesame", "shellfish"):
        with pytest.raises(PosError, match="doesn't track"):
            pos.with_allergen(said)
    with pytest.raises(PosError, match="one allergen per call"):
        pos.with_allergen("nuts and dairy")


def test_every_allergy_mentioned_is_kept_and_flagged(pos: Pos) -> None:
    pos.note_allergy("nuts", severe=True)
    pos.note_allergy("dairy", severe=False)
    pos.note_allergy("nuts", severe=False)  # asked again: still severe
    assert pos.allergies == {"nuts": True, "dairy": False}
    assert pos.snapshot()["allergies"] == [
        {"allergen": "dairy", "severe": False},
        {"allergen": "nuts", "severe": True},
    ]
    assert "dairy" in _line_note(pos, pos.add("cheese", 1, "large"))
    pesto = _line_note(pos, pos.add("pesto", 1, "small"))
    assert "dairy and nuts" in pesto
    assert _line_note(pos, pos.add("house_salad")) == ""
    assert grade(pos, {"allergy": ["nuts", "dairy"]}) == []
    assert grade(pos, {"allergy": "gluten"}) == [
        "the gluten allergy was never looked up in the menu data"
    ]


def test_leaving_a_topping_off(pos: Pos) -> None:
    line = pos.add("veggie", 1, "large", without=["no olives"])
    assert line.describe() == "L1: 1 × large Garden Veggie, no black olives"
    assert line.total == Decimal("19.99")
    pos.update("L1", without=["olives", "hold the onions"])
    assert line.without == ["black olives", "onions"] and line.rev == 2
    pos.update("L1", without=[])
    assert line.without == []
    # the same pizza without something is its own line
    pos.add("bbq_chicken", 1, "medium", without=["onion"])
    assert pos.add("bbq_chicken", 1, "medium").without == []
    assert pos.lines[1].modifiers() == ["no red onions"]
    assert pos.snapshot()["cart"]["lines"][1]["modifiers"] == ["no red onions"]
    with pytest.raises(PosError, match="doesn't come with 'olives'"):
        pos.add("pepperoni", 1, "large", without=["olives"])
    with pytest.raises(PosError, match="leave off"):
        pos.add("pepperoni", 1, "large", toppings=["no olives"])
    with pytest.raises(PosError, match="added topping"):
        pos.add("veggie", 1, "small", toppings=["olives"], without=["olives"])
    with pytest.raises(PosError, match="Only pizzas take changes"):
        pos.add("caesar", without=["croutons"])
    # "a veggie, no olives, and wings": the olives were never on the wings
    with pytest.raises(PosError, match="put it on that pizza's line") as e:
        pos.add("wings", 1, "dozen", sauce="hot", without=["olives"])
    assert "comes with" not in str(e.value)


def test_half_and_half_is_one_pizza_with_two_halves(pos: Pos) -> None:
    line = pos.add("cheese", 1, "large", first_half=["mushroom"], second_half=["pepperoni"])
    assert line.describe() == "L1: 1 × large Cheese, half mushrooms, other half pepperoni"
    # a half topping is half a topping's price
    assert line.unit_price == line.size.price + Decimal("2.00")
    assert pos.snapshot()["cart"]["lines"][0]["modifiers"] == [
        "half mushrooms",
        "other half pepperoni",
    ]
    # the same pizza said again, halves either way round, is a repeat
    with pytest.raises(PosError, match="already in the cart"):
        pos.add("cheese", 1, "large", first_half=["pepperoni"], second_half=["mushrooms"])
    pos.update("L1", second_half=["sausage"])
    assert pos.lines[0].modifiers() == ["half mushrooms", "other half sausage"]
    assert pos.lines[0].was == "1 × large Cheese, half mushrooms, other half pepperoni"
    with pytest.raises(PosError, match="on both halves"):
        pos.update("L1", second_half=["mushrooms"])
    with pytest.raises(PosError, match="already on the whole pizza"):
        pos.update("L1", toppings=["mushrooms"])
    with pytest.raises(PosError, match="can't be half-and-half"):
        pos.add("wings", 1, "dozen", sauce="hot", first_half=["mushrooms"])


def test_dipping_sauces_are_on_the_menu(pos: Pos) -> None:
    assert pos.add("garlic_sauce", 2).describe() == "L1: 2 × Garlic Sauce"
    assert pos.item("ranch").category == "dips"
    assert pos.subtotal() == Decimal("1.98")


def test_ro07_repeating_an_item_never_makes_a_second_line(pos: Pos) -> None:
    pos.add("pepperoni", 2, "large")
    with pytest.raises(PosError, match="already in the cart"):
        pos.add("pepperoni", 2, "large")
    # the same pizza with a different topping is a different line
    pos.add("pepperoni", 1, "large", toppings=["mushrooms"])
    # "yeah, two" again as an update is a no-op, and doesn't invalidate a quote
    _ready_for_pickup(pos)
    pos.price()
    pos.update("L1", quantity=2, size="large")
    assert pos.quote_current
    assert [ln.quantity for ln in pos.lines] == [2, 1]
    # "yeah, two" read as two more is refused: an increase has to be asked for
    with pytest.raises(PosError, match="already has 2"):
        pos.update("L1", quantity=4)
    pos.update("L1", quantity=3, more=True)
    pos.update("L2", quantity=1)
    assert [ln.quantity for ln in pos.lines] == [3, 1]


def test_required_modifiers(pos: Pos) -> None:
    with pytest.raises(PosError, match="needs a size"):
        pos.add("margherita")
    with pytest.raises(PosError, match="need a sauce"):
        pos.add("wings", 1, "dozen")
    with pytest.raises(PosError, match="needs a size"):
        pos.add("sprite")
    with pytest.raises(PosError, match="comes in"):
        pos.add("margherita", 1, "extra large")
    with pytest.raises(PosError, match="no sauce called"):
        pos.add("wings", 1, "dozen", sauce="ranch")
    with pytest.raises(PosError, match="doesn't take toppings"):
        pos.add("caesar", toppings=["bacon"])
    with pytest.raises(PosError, match="isn't on the menu"):
        pos.add("calzone")
    # a one-size item needs nothing
    assert pos.add("breadsticks").size.price == Decimal("5.99")


def test_a_dozen_wings_is_one_order(pos: Pos) -> None:
    with pytest.raises(PosError, match="use quantity 1"):
        pos.add("wings", 12, "dozen", sauce="hot")
    line = pos.add("wings", 1, "dozen", sauce="hot")
    with pytest.raises(PosError, match="use quantity 1"):
        pos.update(line.id, quantity=6, size="half dozen", more=True)


def test_toppings_are_priced(pos: Pos) -> None:
    line = pos.add("margherita", 2, "M", toppings=["Jalapeños", "extra cheese"])
    assert line.unit_price == Decimal("15.99") + Decimal("4.00")
    assert line.total == Decimal("39.98")


def test_submit_needs_a_read_back_and_details(pos: Pos) -> None:
    with pytest.raises(PosError, match="empty"):
        pos.submit()
    pos.add("pepperoni", 2, "large")
    with pytest.raises(PosError, match="Pickup or delivery"):
        pos.submit()
    with pytest.raises(PosError, match="pickup or delivery"):
        pos.price()
    pos.set_pickup()
    with pytest.raises(PosError, match="name"):
        pos.submit()
    pos.set_contact("Sam", "415 555 0142")
    with pytest.raises(PosError, match="read it back"):
        pos.submit()
    pos.price()
    # switching to delivery changes the total, so the read-back is stale
    pos.set_delivery("1355 Market St, 94103")
    with pytest.raises(PosError, match="changed after"):
        pos.submit()
    pos.price()
    pos.submit()
    with pytest.raises(PosError, match="already sent"):
        pos.add("coke", 1, "small")


def test_update_can_change_the_item_and_keeps_its_size(pos: Pos) -> None:
    pos.add("coke", 2, "large")
    line = pos.update("L1", item_id="sprite")
    assert line.describe() == "L1: 2 × large Sprite" and line.was == "2 × large Coke"
    # lemonade comes regular or large: the large carries over
    assert pos.update("L1", item_id="lemonade").describe() == "L1: 2 × large Lemonade"
    pizza = pos.add("pepperoni", 1, "medium", toppings=["mushrooms"], without=["pepperoni"])
    # a new pizza keeps the size and the added toppings; what was left off was the old one's
    line = pos.update(pizza.id, item_id="cheese")
    assert line.describe() == "L2: 1 × medium Cheese, + mushrooms"
    wings = pos.add("wings", 1, "dozen", sauce="BBQ")
    assert pos.update(wings.id, item_id="boneless").describe() == (
        "L3: 1 × dozen Boneless Wings, BBQ sauce"
    )
    # a one-size item drops the size; a size the new item doesn't come in is asked
    assert pos.update(wings.id, item_id="breadsticks").describe() == "L3: 1 × Breadsticks"
    pos.add("water")
    with pytest.raises(PosError, match="needs a size"):
        pos.update("L4", item_id="coke")
    with pytest.raises(PosError, match="sold out"):
        pos.update("L4", item_id="buffalo_cauliflower")
    assert pos.update("L4", item_id="coke", size="small").describe() == "L4: 1 × small Coke"


def test_upsell_at_most_once(pos: Pos) -> None:
    pos.offer_upsell()
    with pytest.raises(PosError, match="already offered"):
        pos.offer_upsell()
    fresh = demo_pos()
    fresh.add("breadsticks")
    with pytest.raises(PosError, match="already in the order"):
        fresh.offer_upsell()


def test_bad_phone_is_refused(pos: Pos) -> None:
    with pytest.raises(PosError, match="ten-digit"):
        pos.set_contact(phone="555-0142")
    pos.set_contact(phone="1 415 555 0142")
    assert pos.phone == "(415) 555-0142"


@pytest.mark.parametrize(
    "said", ["494", "123 456 7890", "415 111 0142", "911 555 0142", "000 000 0000"]
)
def test_a_number_that_cant_be_real_is_refused(pos: Pos, said: str) -> None:
    with pytest.raises(PosError, match="real ten-digit"):
        pos.set_contact(phone=said)
    assert pos.phone is None


@pytest.mark.parametrize(
    "said",
    [
        "415-555-0142",
        "(415) 555 0142",
        "four one five, five five five, oh one four two",
        "4 1 5 5 5 5 zero one 4 2",
        "415 555 o142",
        "my cell is 415 555 0142",
    ],
)
def test_a_phone_number_said_any_way(pos: Pos, said: str) -> None:
    pos.set_contact(phone=said)
    assert pos.phone == "(415) 555-0142"


def test_natural_names_for_items_sizes_and_sauces(pos: Pos) -> None:
    # "hot sauce", "a large", "pepperoni pizza", "meat lover's": what callers (and the
    # model, copying them) say, not the menu's exact words
    assert pos.item("pepperoni pizza").id == "pepperoni"
    assert pos.item("Meat Lover's pizza").id == "meat_lovers"
    assert pos.item("sausage and peppers").id == "sausage_peppers"
    assert pos.item("breadstick").id == "breadsticks"
    assert pos.item("cokes").id == "coke"
    assert pos.item("boneless").id == "boneless"
    line = pos.add("wings", 1, "12 piece", sauce="hot sauce")
    assert line.describe() == "L1: 1 × dozen Bone-in Wings, hot sauce"
    assert pos.add("margherita", 1, "a large", toppings=["extra mushrooms"]).describe() == (
        "L2: 1 × large Margherita, + mushrooms"
    )
    assert pos.add("coke", 2, "large").size.id == "L"
    assert pos.add("boneless", 1, "six pieces", sauce="BBQ sauce").size.id == "6"
    # still refused: not on the menu, not a size, not a sauce
    with pytest.raises(PosError, match="isn't on the menu"):
        pos.item("pizza")
    with pytest.raises(PosError, match="comes in"):
        pos.add("cheese", 1, "extra large")
    with pytest.raises(PosError, match="no sauce called"):
        pos.add("wings", 1, "dozen", sauce="ranch sauce")


def test_the_model_sees_opaque_menu_ids(pos: Pos) -> None:
    refs = [m.ref for m in pos.menu]
    assert len(set(refs)) == len(refs)
    assert not set(refs) & {m.id for m in pos.menu}
    assert all(pos.item(m.ref) is m for m in pos.menu)
    wings = pos.item("wings")
    assert (
        wings.describe()
        == f"{wings.ref}: Bone-in Wings (half dozen $8.99, dozen $16.99). Sauce required."
    )


def test_grade_passes_the_expected_order(pos: Pos) -> None:
    pos.add("pepperoni", 1, "large")
    pos.add("wings", 1, "dozen", sauce="hot")
    _ready_for_pickup(pos)
    pos.price()
    pos.submit()
    expect = {
        "submitted": True,
        "fulfillment": "pickup",
        "cart": ["1 × dozen Bone-in Wings, hot sauce", "1 × large Pepperoni"],
        "total": "40.31",
        "name": "Sam Carter",
        "phone": "(415) 555-0142",
    }
    assert grade(pos, expect) == []


def test_grade_flags_a_wrong_cart_and_a_missing_order(pos: Pos) -> None:
    pos.add("pepperoni", 2, "large")
    pos.add("coke", 1, "small")
    _ready_for_pickup(pos)
    pos.price()
    problems = grade(
        pos,
        {"submitted": True, "cart": ["1 × large Pepperoni"], "not_in_cart": ["coke"]},
    )
    assert len(problems) == 3
    assert problems[0].startswith("no order was submitted")


def test_grade_flags_an_order_that_mustnt_be_submitted(pos: Pos) -> None:
    pos.add("cheese", 1, "small")
    _ready_for_pickup(pos)
    pos.price()
    pos.submit()
    assert grade(pos, {"submitted": False}) == [
        f"order #{pos.order.number} was submitted but shouldn't have been"  # type: ignore[union-attr]
    ]


def test_a_tip_goes_on_the_total_and_needs_a_new_read_back(pos: Pos) -> None:
    pos.add("pepperoni", 1, "large")
    pos.add("wings", 1, "dozen", sauce="hot")
    _ready_for_pickup(pos)
    pos.price()
    tip = pos.set_tip(Decimal("5"))
    assert tip == Decimal("5.00") and not pos.quote_current
    with pytest.raises(PosError, match="changed after it was read back"):
        pos.submit()
    q = pos.price()
    assert (q.tip, q.total) == (Decimal("5.00"), Decimal("45.31"))
    assert "tip $5.00" in q.describe()
    assert pos.snapshot()["quote"]["tip"] == 5.0
    # the same tip again is no change; a percent is of the subtotal ($36.98), to the cent
    pos.set_tip(Decimal("5.00"))
    assert pos.quote_current
    assert pos.set_tip(percent=Decimal("15")) == Decimal("5.55")
    for bad in ({"amount": Decimal("-1")}, {"amount": Decimal("101")}):
        with pytest.raises(PosError, match="between"):
            pos.set_tip(**bad)
    with pytest.raises(PosError, match="one of the two"):
        pos.set_tip(Decimal("5"), Decimal("10"))
    pos.price()
    pos.submit()
    assert grade(pos, {"submitted": True, "tip": "5.55", "total": "45.86"}) == []
    with pytest.raises(PosError, match="already sent"):
        pos.set_tip(Decimal("10"))


def test_payment_is_in_person(pos: Pos) -> None:
    assert pos.payment() is None and pos.snapshot()["payment"] is None
    pos.set_pickup()
    assert pos.payment() == "Pay at pickup, cash or card"
    pos.set_delivery("300 De Haro St, 94107")
    assert pos.snapshot()["payment"] == "Pay the driver at the door, cash or card"


def test_cancelling_before_submit_places_nothing(pos: Pos) -> None:
    pos.add("pepperoni", 1, "large")
    assert pos.cancel() == "not_placed" and pos.cancel(confirmed=True) == "not_placed"
    assert pos.order is None
    assert grade(pos, {"submitted": False, "cancelled": False}) == []


def test_the_order_placed_on_the_call_can_be_cancelled_after_a_read_back(pos: Pos) -> None:
    pos.add("pepperoni", 1, "large")
    _ready_for_pickup(pos)
    pos.price()
    order = pos.submit()
    assert pos.cancel(confirmed=True) == "ask" and not order.cancelled
    assert pos.cancel(confirmed=True) == "cancelled"
    assert order.cancelled
    snap = pos.snapshot()
    assert snap["order"]["cancelled"] is True
    assert snap["payment"] == "Nothing due: the order was cancelled"
    with pytest.raises(PosError, match="already cancelled"):
        pos.cancel(confirmed=True)
    with pytest.raises(PosError, match="cancelled on this call"):
        pos.add("coke", 1, "small")
    with pytest.raises(PosError, match="cancelled on this call"):
        pos.set_tip(Decimal("2"))
    assert grade(pos, {"submitted": True, "cancelled": True}) == []
    assert grade(pos, {"cancelled": False}) == ["cancelled is True, expected False"]


def test_a_delivery_note_for_the_driver(pos: Pos) -> None:
    pos.add("pepperoni", 1, "large")
    pos.set_pickup()
    with pytest.raises(PosError, match="no driver"):
        pos.set_delivery_note("Leave it at the door")
    pos.set_delivery("123 Market St, 94105")
    pos.set_contact("Sam", "415 555 0142")
    pos.price()
    for bad, why in (("   ", "empty"), ("x" * 201, "keep it under 200")):
        with pytest.raises(PosError, match=why):
            pos.set_delivery_note(bad)
    note = pos.set_delivery_note("Leave it at the door,  don't ring the bell")
    assert note == "Leave it at the door, don't ring the bell"
    # it's read back with the order, so a new note needs a new read-back
    assert not pos.quote_current
    assert pos.snapshot()["fulfillment"]["note"] == note
    # a corrected address keeps it
    pos.set_delivery("125 Market St, 94105")
    assert pos.fulfillment.note == note
    pos.price()
    order = pos.submit()
    assert order.note == note
    assert grade(pos, {"driver_note": ["door", "bell"]}) == []
    assert grade(pos, {"driver_note": ["gate"]}) == [f"driver note {note!r} is missing 'gate'"]
    with pytest.raises(PosError, match="already sent"):
        pos.set_delivery_note("Call on arrival")


@pytest.mark.parametrize(
    "note",
    ["Deliver to 500 Howard Street", "bring it to 1355 Market St instead", "455 natoma st"],
)
def test_a_note_that_names_an_address_is_refused(pos: Pos, out: Pos, note: str) -> None:
    assert looks_like_an_address(note)
    pos.add("pepperoni", 1, "large")
    pos.set_delivery("123 Market St, 94105")
    with pytest.raises(PosError, match="call set_delivery"):
        pos.set_delivery_note(note)
    # on its way, the order goes to the address on it
    with pytest.raises(PosError, match="can't change where the order goes"):
        out.set_driver_note(note)
    assert out.delivery is not None and out.delivery.note is None


@pytest.mark.parametrize(
    "note",
    ["Gate code 4512, unit 3B", "Leave it at the door, don't ring the bell", "Call on arrival"],
)
def test_drop_off_details_are_not_an_address(note: str) -> None:
    assert not looks_like_an_address(note)


# ---- the delivery-update call ----


@pytest.fixture()
def out() -> Pos:
    return demo_pos("outbound", "Sam Carter")


def test_outbound_seeds_order_214_out_for_delivery(out: Pos) -> None:
    assert out.order is not None and out.order.number == "214"
    assert out.delivery is not None and out.delivery.due == Decimal("49.65")
    assert out.payment() == "Pay the driver at the door, cash or card"
    assert out.fulfillment.zip == "94107"
    assert [ln.item.id for ln in out.out_of_stock()] == ["sprite"]
    assert out.same_price_swaps(out.line("L3"))[0] == "large Coke"
    # the inbound demo still starts empty, with the Sprite in stock
    assert demo_pos().item("sprite").available and not demo_pos().lines


def test_the_opener_goes_straight_into_the_update(out: Pos) -> None:
    assert _opener(out) == (
        "Hi, Sam, this is Gabby from Zappi's Pizzeria, calling about your delivery order. "
        "Marco's on the way with it, about 5 minutes out, but the Sprite ran out after it went "
        "in. Would a large Coke at the same price work instead, or should I take it off?"
    )
    said = str(DeliveryUpdateAgent(out).instructions)
    assert "outbound call to\nSam Carter" in said and "never ask\nwho they are" in said
    # whoever picks up is the customer: no question about who they are, to open or after
    for spoken in (_opener(out), said):
        assert "speaking with" not in spoken.lower() and "Did you place" not in spoken
        assert "Once they say it's them" not in spoken


def test_without_a_name_nobody_is_named() -> None:
    pos = demo_pos("outbound")
    assert not pos.name and pos.snapshot()["customer"]["name"] == "Customer"
    assert _opener(pos) == (
        "Hi, this is Gabby from Zappi's Pizzeria, calling about your delivery order. Marco's on "
        "the way with it, about 5 minutes out, but the Sprite ran out after it went in. Would a "
        "large Coke at the same price work instead, or should I take it off?"
    )
    said = str(DeliveryUpdateAgent(pos).instructions)
    assert "the customer who placed it" in said and "never say or guess one" in said
    assert "Did you place" not in _opener(pos) and "never ask\nwho they are" in said
    for spoken in (_opener(pos), said, _voicemail()):
        assert "Sam" not in spoken and "Carter" not in spoken
        assert "with there" not in spoken and "for there" not in spoken and "there," not in spoken
    # the inbound order has no name until the caller gives one, and the page shows none
    assert demo_pos().snapshot()["customer"]["name"] is None


def test_the_voicemail_names_nobody() -> None:
    assert _voicemail().startswith(
        "Hi, this is Gabby from Zappi's Pizzeria, with a quick update on your delivery."
    )


def test_swap_keeps_the_price(out: Pos) -> None:
    line = out.swap_out_of_stock("L3", "coke")
    assert line.describe() == "L3: 1 × large Coke" and line.was == "1 × large Sprite"
    assert out.quote is not None and out.quote.total == Decimal("49.65")
    assert out.balance() == 0 and out.quote_current
    assert grade(out, {"cart": ["1 × large Pepperoni"]})  # not read back, cart differs


def test_swap_must_match_the_category_and_price(out: Pos) -> None:
    with pytest.raises(PosError, match="same price"):
        out.swap_out_of_stock("L3", "cannoli")
    with pytest.raises(PosError, match="doesn't come at"):
        out.swap_out_of_stock("L3", "water")
    with pytest.raises(PosError, match="isn't out of stock"):
        out.swap_out_of_stock("L1", "cheese")


def test_taking_it_off_makes_the_item_and_its_tax_less_at_the_door(out: Pos) -> None:
    _, less = out.remove_out_of_stock("L3")
    assert less == Decimal("4.35")
    assert out.quote is not None and out.quote.total == Decimal("45.30")
    assert out.balance() == Decimal("-4.35") and out.due_now() == Decimal("45.30")
    assert out.delivery is not None
    assert out.delivery.changes["swap"] == "Removed 1 × large Sprite, $4.35 less at the door"
    snap = out.snapshot()["delivery"]
    assert (snap["due"], snap["dueNow"], snap["balance"]) == (49.65, 45.30, -4.35)
    assert [ln.item.id for ln in out.removed] == ["sprite"]


def test_only_breadsticks_can_ride_along(out: Pos) -> None:
    with pytest.raises(PosError, match="can't be added now"):
        out.add_extra("cannoli")
    with pytest.raises(PosError, match="at most"):
        out.add_extra("breadsticks", 3)
    _, charge = out.add_extra("breadsticks")
    assert charge == Decimal("6.53")
    # saying it again is a repeat, not a second order
    assert out.add_extra("breadsticks") == (out.lines[-1], Decimal("0"))
    assert out.lines[-1].quantity == 1
    # the rest of the order is locked, as before
    with pytest.raises(PosError, match="already sent"):
        out.add("cannoli")


def test_the_order_out_with_the_driver_can_be_cancelled_after_a_read_back(out: Pos) -> None:
    out.add_extra("breadsticks")
    # the first call reads it back; nothing is cancelled until a confirmed call after that
    assert out.cancel(confirmed=True) == "ask"
    assert out.order is not None and not out.order.cancelled
    # a change in between needs the cancellation read back again
    out.set_driver_note("Leave at the door")
    assert not out.cancel_asked
    assert out.cancel() == "ask"
    assert out.cancel(confirmed=True) == "cancelled"
    assert out.order.cancelled and out.due_now() == 0
    assert out.balance() == Decimal("-49.65")  # nothing of the $49.65 due when it went out
    assert out.delivery is not None
    assert list(out.delivery.changes.values()) == ["Cancelled order 214; nothing is due"]
    # the Sprite needn't be resolved: the read-back is the cancellation
    out.read_back()
    assert out.call_done
    snap = out.snapshot()
    assert snap["order"]["cancelled"] and snap["delivery"]["cancelled"]
    assert snap["delivery"]["dueNow"] == 0
    assert snap["payment"] == "Nothing due: the order was cancelled"
    for act in (
        lambda: out.swap_out_of_stock("L3", "coke"),
        lambda: out.add_extra("breadsticks", 2),
        lambda: out.set_driver_note("Gate 4512"),
        lambda: out.set_tip(Decimal("5")),
    ):
        with pytest.raises(PosError, match="cancelled on this call"):
            act()
    with pytest.raises(PosError, match="cancelled on this call"):
        out.cancel()
    assert grade(out, {"cancelled": True, "balance": "-49.65"}) == []
    assert grade(out, {"cancelled": False}) == ["cancelled is True, expected False"]


def test_a_tip_for_the_driver_is_added_at_the_door(out: Pos) -> None:
    out.swap_out_of_stock("L3", "coke")
    out.read_back()
    assert out.set_tip(Decimal("5")) == Decimal("5.00")
    assert out.quote is not None and out.quote.total == Decimal("54.65")
    assert out.balance() == Decimal("5.00") and not out.read_back_current
    assert out.delivery is not None
    assert out.delivery.changes["tip"] == "Tip for Marco $5.00, added at the door"
    # 10% of the $40.97 subtotal, to the cent; 0 takes it off the read-back
    assert out.set_tip(percent=Decimal("10")) == Decimal("4.10")
    assert out.snapshot()["quote"]["tip"] == 4.10
    out.set_tip(Decimal("0"))
    assert "tip" not in out.delivery.changes and out.balance() == 0
    out.set_tip(Decimal("3"))
    out.read_back()
    assert grade(out, {"tip": "3", "balance": "3"}) == []


def test_driver_note_replaces_and_is_checked(out: Pos) -> None:
    with pytest.raises(PosError, match="empty"):
        out.set_driver_note("   ")
    with pytest.raises(PosError, match="under"):
        out.set_driver_note("x" * 201)
    out.set_driver_note("Gate code 4512")
    out.set_driver_note("Gate code 4512,  unit 3B")
    assert out.delivery is not None and out.delivery.note == "Gate code 4512, unit 3B"
    assert out.snapshot()["delivery"]["note"] == "Gate code 4512, unit 3B"


def test_read_back_needs_the_sprite_resolved_and_goes_stale(out: Pos) -> None:
    with pytest.raises(PosError, match="still unresolved"):
        out.read_back()
    out.swap_out_of_stock("L3", "coke")
    out.read_back()
    assert out.read_back_current
    out.set_driver_note("Leave at the door")
    assert not out.read_back_current
    out.read_back()
    expect = {
        "cart": ["1 × large Pepperoni", "1 × dozen Bone-in Wings, hot sauce", "1 × large Coke"],
        "total": "49.65",
        "balance": "0",
        "driver_note": ["door"],
        "cancelled": False,
        "tip": "0",
    }
    assert grade(out, expect) == []


def test_grade_flags_an_unresolved_outbound_call(out: Pos) -> None:
    problems = grade(out, {})
    assert any("out of stock" in p for p in problems)
    assert any("read back" in p for p in problems)


def test_the_sprite_choice_can_change_after_a_swap(out: Pos) -> None:
    out.swap_out_of_stock("L3", "coke")
    line = out.swap_out_of_stock("L3", "diet_coke")
    assert line.describe() == "L3: 1 × large Diet Coke" and line.was == "1 × large Coke"
    assert out.balance() == 0
    assert out.delivery is not None
    swap = out.delivery.changes["swap"]
    assert swap == "1 × large Sprite → 1 × large Diet Coke, same price"
    # "actually, just take it off": it comes off as the Sprite that was ordered
    _, less = out.remove_out_of_stock("L3")
    assert less == Decimal("4.35") and out.balance() == Decimal("-4.35")
    assert [ln.describe() for ln in out.removed] == ["L3: 1 × large Sprite"]
    assert out.delivery.changes["swap"] == "Removed 1 × large Sprite, $4.35 less at the door"
    with pytest.raises(PosError, match="already taken off"):
        out.remove_out_of_stock("L3")
    # the rest of the order still can't change
    with pytest.raises(PosError, match="isn't out of stock"):
        out.swap_out_of_stock("L1", "cheese")


def test_a_sprite_taken_off_can_become_a_coke(out: Pos) -> None:
    out.add_extra("breadsticks")
    out.remove_out_of_stock("L3")
    line = out.swap_out_of_stock("l3", "coke")
    assert [ln.id for ln in out.lines] == ["L1", "L2", "L3", "L4"] and not out.removed
    assert line.describe() == "L3: 1 × large Coke"
    assert line.was == "1 × large Sprite"
    assert out.balance() == Decimal("6.53")  # the breadsticks, and the Sprite's price back
    out.read_back()
    expect = {
        "cart": [
            "1 × large Pepperoni",
            "1 × dozen Bone-in Wings, hot sauce",
            "1 × large Coke",
            "1 × Breadsticks",
        ],
        "not_in_cart": ["sprite"],
        "balance": "6.53",
    }
    assert grade(out, expect) == []


def test_breadsticks_can_be_recounted_or_taken_off(out: Pos) -> None:
    out.swap_out_of_stock("L3", "coke")
    out.add_extra("breadsticks")
    line, charge = out.add_extra("breadsticks", 2)
    assert line.quantity == 2 and line.was == "1 × Breadsticks"
    assert charge == Decimal("6.53") and out.balance() == Decimal("13.06")
    assert out.quote is not None and out.quote.total == Decimal("62.71")
    assert out.delivery is not None
    assert out.delivery.changes["extra"] == "Added 2 × Breadsticks, $13.06 more at the door"
    with pytest.raises(PosError, match="at most 2"):
        out.add_extra("breadsticks", 3)
    out.read_back()
    _, charge = out.add_extra("breadsticks", 0)
    assert charge == Decimal("-13.06") and out.balance() == 0
    assert "extra" not in out.delivery.changes and not out.read_back_current
    assert not any(ln.item.id == "breadsticks" for ln in out.lines + out.removed)
    with pytest.raises(PosError, match="no Breadsticks"):
        out.add_extra("breadsticks", 0)


def test_wrong_person_ends_the_call_with_nothing_shared_or_changed(out: Pos) -> None:
    assert not out.call_done
    out.end_without_update("wrong_person")
    assert out.call_done
    assert out.delivery is not None and out.delivery.ended_early == "wrong_person"
    assert out.snapshot()["delivery"]["endedEarly"] == "Wrong person"
    # nothing about the order goes to whoever answered
    for act in (
        lambda: out.swap_out_of_stock("L3", "coke"),
        lambda: out.add_extra("breadsticks"),
        lambda: out.set_driver_note("Leave at the door"),
        lambda: out.set_tip(Decimal("5")),
        out.cancel,
        out.read_back,
    ):
        with pytest.raises(PosError, match="isn't the customer"):
            act()
    assert out.order is not None and not out.order.cancelled and not out.cancel_asked
    assert grade(out, {"ended_early": "wrong_person", "unchanged": True}) == []
    assert grade(out, {}) == ["the call ended early as wrong_person, expected None"]


def test_call_back_later_can_end_the_call_until_they_change_something(out: Pos) -> None:
    out.end_without_update("call_back_later")
    assert out.call_done
    # they carry on after all: now it's a normal update, read back before it ends
    out.swap_out_of_stock("L3", "coke")
    assert out.delivery is not None and out.delivery.ended_early is None
    assert not out.call_done
    out.read_back()
    assert out.call_done


def test_snapshot_is_what_the_page_draws(pos: Pos) -> None:
    pos.add("wings", 1, "dozen", sauce="hot")
    pos.set_pickup()
    pos.price()
    snap = pos.snapshot()
    wings = next(m for m in snap["menu"] if m["id"] == "wings")
    assert wings["sauceAllergens"]["Thai chili peanut"] == ["nuts"]
    cauli = next(m for m in snap["menu"] if m["id"] == "buffalo_cauliflower")
    assert cauli["available"] is False
    assert snap["cart"]["lines"][0]["modifiers"] == ["hot sauce"]
    assert snap["quote"]["current"] is True
    assert snap["quote"]["total"] == 18.52


def test_the_page_shows_wings_by_the_count(pos: Pos) -> None:
    pos.add("wings", 1, "dozen", sauce="hot")
    pos.add("boneless", 1, "half dozen", sauce="BBQ")
    pos.add("pepperoni", 1, "large")
    sizes = [ln["size"] for ln in pos.snapshot()["cart"]["lines"]]
    # "12 Bone-in Wings" on the page, while the agent still says "a dozen"
    assert sizes == ["12", "6", "large"]
    assert pos.lines[0].describe() == "L1: 1 × dozen Bone-in Wings, hot sauce"


def test_a_soda_comes_small_medium_or_large(pos: Pos) -> None:
    # "a large pepperoni and a Coke": the Coke has no size until the caller gives one
    with pytest.raises(PosError, match="Coke needs a size: small, medium, or large. Ask"):
        pos.add("coke")
    with pytest.raises(PosError, match="Coke comes in small, medium, or large, not '2 liter'"):
        pos.add("coke", 1, "2 liter")
    assert not pos.lines
    assert [
        pos.add(d, 1, s).size.price
        for d, s in [("coke", "a small"), ("sprite", "medium"), ("diet_coke", "large")]
    ] == [Decimal("1.99"), Decimal("2.99"), Decimal("3.99")]


def _said(*turns: tuple[str, str]) -> Any:
    chat = llm.ChatContext()
    for role, text in turns:
        chat.add_message(role=role, content=text)  # type: ignore[arg-type]
    return SimpleNamespace(session=SimpleNamespace(history=chat))


@pytest.mark.parametrize(
    "turns",
    [
        [("user", "Can I get another pepperoni?")],
        [("user", "Make it three.")],
        [("user", "No, the other pizza. Put it back.")],
        [("assistant", "Want to add one more large pepperoni?"), ("user", "Yes")],
        [("assistant", "Should I add another one?"), ("user", "yeah, please")],
    ],
)
def test_asking_for_more_raises_the_count(turns: list[tuple[str, str]]) -> None:
    assert _asked_for_more(_said(*turns), 3)


@pytest.mark.parametrize(
    "turns",
    [
        [("user", "Yeah, two pepperonis.")],
        [("assistant", "Anything else?"), ("user", "Yes")],
        [("assistant", "Want to add one more pepperoni?"), ("user", "No thanks")],
    ],
)
def test_repeating_or_a_plain_yes_keeps_the_count(turns: list[tuple[str, str]]) -> None:
    assert not _asked_for_more(_said(*turns), 3)


@pytest.mark.parametrize(
    "turns",
    [
        [("user", "A large Coke, please.")],
        [("user", "Two cokes, small.")],
        [("user", "Two Cokes."), ("assistant", "What size?"), ("user", "Small. Small is fine.")],
        [("user", "Medium for the Coke, and a large pepperoni.")],
        [("user", "And a large soda, Coke.")],
    ],
)
def test_a_drink_size_the_caller_gave_is_taken(pos: Pos, turns: list[tuple[str, str]]) -> None:
    size = next(w for w in ("small", "medium", "large") if w in str(turns).lower())
    _drink_size_said(_said(*turns), pos, "coke", size)


def test_a_drinks_size_carries_to_the_drink_it_becomes(pos: Pos) -> None:
    _drink_size_said(_said(("user", "Make the large Coke a Sprite.")), pos, "sprite", "large")


@pytest.mark.parametrize(
    ("said", "size"),
    [
        ("Large pepperoni and a Coke.", "large"),
        ("A large pizza and a coke", "large"),
        ("Large pepperoni and a Coke.", "small"),
        ("Yeah, a Coke.", "medium"),
    ],
)
def test_a_drink_size_is_never_borrowed_or_picked(pos: Pos, said: str, size: str) -> None:
    with pytest.raises(PosError, match="hasn't said a size for the Coke"):
        _drink_size_said(_said(("user", said)), pos, "coke", size)


def _heard_at(*turns: tuple[str, str, float]) -> Any:
    chat = llm.ChatContext()
    for role, text, at in turns:
        chat.add_message(role=role, content=text, created_at=at)  # type: ignore[arg-type]
    return SimpleNamespace(session=SimpleNamespace(history=chat))


@pytest.mark.parametrize("answer", ["Yes.", "Yeah, cancel it.", "Do it", "please"])
def test_a_cancel_goes_through_on_a_yes_after_the_read_back(answer: str) -> None:
    ctx = _heard_at(("user", "Cancel my order.", 1.0), ("user", answer, 3.0))
    assert _answered_yes_since(ctx, 2.0)


@pytest.mark.parametrize(
    ("turns", "asked_at"),
    [
        # the read-back was never given, or the caller hasn't answered it yet
        ([("user", "Cancel my order.", 1.0)], None),
        ([("user", "Cancel my order.", 1.0)], 2.0),
        # an answer, but a no
        ([("user", "No, wait, keep it.", 3.0)], 2.0),
        ([("user", "Don't cancel it.", 3.0)], 2.0),
        ([("user", "Hold on.", 3.0)], 2.0),
    ],
)
def test_a_cancel_never_goes_through_without_a_yes(
    turns: list[tuple[str, str, float]], asked_at: float | None
) -> None:
    ctx = _heard_at(*turns)
    assert not _answered_yes_since(ctx, asked_at)
    pos = demo_pos("outbound")
    pos.cancel()
    with pytest.raises(PosError, match="haven't said yes"):
        _cancel_needs_a_yes(ctx, pos, True, asked_at)
    # the first call, to read it back, needs no yes
    _cancel_needs_a_yes(ctx, pos, False, asked_at)


def test_a_cancellation_asked_in_her_own_words_counts_as_the_read_back() -> None:
    asked = _heard_at(
        ("user", "Just cancel it.", 1.0),
        ("assistant", "Cancel the whole order? Nothing will be due.", 2.0),
        ("user", "Yes.", 3.0),
    )
    assert _cancel_read_back_at(asked) == 2.0
    # a statement about cancelling, or a later question about something else, isn't one
    for said in ("I can cancel it for you.", "Anything else?"):
        other = _heard_at(
            ("assistant", "Cancel the whole order?", 1.0),
            ("assistant", said, 2.0),
            ("user", "Yes.", 3.0),
        )
        assert _cancel_read_back_at(other) is None


@pytest.mark.parametrize(
    ("answers", "want"),
    [
        (["Yes, that's right."], "yes"),
        (["Yep."], "yes"),
        (["Sounds good."], "yes"),
        # a yes, then a detail that's none of yes or no: still the yes
        (["Yes.", "Four one five, five five five, zero one four two."], "yes"),
        (["Whatever."], "unsure"),
        (["Sure, I guess."], "unsure"),
        (["I don't know."], "unsure"),
        (["No, that's not right."], "no"),
        (["Wait, make it a medium."], "no"),
        (["Hmm."], "none"),
        ([], "none"),
    ],
)
def test_only_a_clear_yes_answers_the_read_back(answers: list[str], want: str) -> None:
    ctx = _heard_at(
        ("user", "Yes, a large pepperoni.", 1.0),  # before the read-back: doesn't count
        ("assistant", "That's $21.79. Is that right?", 2.0),
        *(("user", a, 3.0 + i) for i, a in enumerate(answers)),
    )
    assert _read_back_answer(ctx, 2.0) == want
    assert _read_back_answer(ctx, None) == "none"


class _Speech:
    async def wait_for_playout(self) -> None:
        return None


class _QuietSession:
    """Just enough of AgentSession for hang_up_when_silent: its events, and what it says."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.said: list[str] = []
        self.resets = 0

    def on(self, event: str) -> Any:
        def register(fn: Any) -> Any:
            self.handlers[event] = fn
            return fn

        return register

    def say(self, text: str, **_: Any) -> _Speech:
        self.said.append(text)
        return _Speech()

    def reset_away_timer(self) -> None:
        self.resets += 1

    async def user(self, old: str, new: str) -> None:
        ev = UserStateChangedEvent(old_state=old, new_state=new)
        self.handlers["user_state_changed"](ev)
        for _ in range(3):
            await asyncio.sleep(0)


async def test_silence_gets_one_check_in_then_a_goodbye() -> None:
    session = _QuietSession()
    page = Page("Gabby", demo_pos())
    hang_up_when_silent(session, page)  # type: ignore[arg-type]
    await session.user("listening", "away")
    assert session.said == [STILL_THERE] and session.resets == 1
    assert page.end_reason is None
    # back to listening after the check-in isn't the caller speaking: the count holds
    await session.user("away", "listening")
    await session.user("listening", "away")
    assert session.said == [STILL_THERE, SILENT_GOODBYE]
    assert page.end_reason == "No response"
    # once it's hung up, more silence does nothing
    await session.user("listening", "away")
    assert session.said == [STILL_THERE, SILENT_GOODBYE]


async def test_the_caller_speaking_resets_the_silence_count() -> None:
    session = _QuietSession()
    page = Page("Gabby", demo_pos())
    hang_up_when_silent(session, page)  # type: ignore[arg-type]
    await session.user("listening", "away")
    await session.user("away", "speaking")
    await session.user("speaking", "listening")
    await session.user("listening", "away")
    # a fresh quiet spell: another check-in, not a hang-up
    assert session.said == [STILL_THERE, STILL_THERE] and page.end_reason is None


def _outbound_said(*turns: tuple[str, str]) -> Any:
    ctx = _said(*turns)
    ctx.userdata = SimpleNamespace(pos=demo_pos("outbound"))
    return ctx


def test_drop_off_details_must_be_saved_before_the_wrap_up() -> None:
    ctx = _outbound_said(("user", "No, just leave it at the door, please."))
    assert _drop_off_unsaved(ctx) == "No, just leave it at the door, please."
    ctx.userdata.pos.set_driver_note("Leave at the door")
    assert _drop_off_unsaved(ctx) is None
    # a discount code isn't a drop-off detail
    assert _drop_off_unsaved(_outbound_said(("user", "Can you use code save twenty?"))) is None


def test_whats_due_doesnt_repeat_how_they_pay() -> None:
    assert "cash or card" not in _money(demo_pos("outbound"))


# ---- the call helpers ----


async def test_pepperonis_are_said_as_pepperoni_pizzas() -> None:
    async def chunks() -> AsyncIterator[str]:
        for c in ["Two large pepper", "onis and a ", "Pepperonis", "."]:
            yield c

    out = "".join([c async for c in say_pepperoni_pizzas(chunks())])
    assert out == "Two large pepperoni pizzas and a Pepperoni pizzas."
    one = "".join([c async for c in say_pepperoni_pizzas(_one("A pepperoni, got it."))])
    assert one == "A pepperoni, got it."


async def _one(text: str) -> AsyncIterator[str]:
    yield text


def test_spoken_number() -> None:
    assert spoken_number("+14155550142") == "415-555-0142"
    assert spoken_number("555-0142") == "555-0142"
