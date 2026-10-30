"""The point of sale on its own: the restaurant's rules, prices, and what the page is sent."""

from decimal import Decimal

import pytest

from pizzeria import Pizzeria, PizzeriaError
from simulation import grade

SAM = {"name": "Sam Carter", "phone": "4155550142"}


def item(item_id, size=None, quantity=1, **kw):
    return {"item_id": item_id, "size": size, "quantity": quantity, **kw}


async def order(*items, mode="pickup", **fulfillment):
    pos = Pizzeria()
    _, problems = await pos.add_items(list(items))
    assert not problems, problems
    await pos.set_fulfillment(mode, **fulfillment)
    await pos.set_contact(**SAM)
    return pos


def total(pos):
    return pos.quote().total


async def test_the_smoke_order_comes_to_its_read_back_total():
    pos = await order(
        item("pepperoni", "large"),
        item("wings", "dozen", sauce="hot"),
        item("coke", "large"),
    )
    assert total(pos) == Decimal("44.66")
    assert sorted(ln.describe() for ln in pos.lines) == [
        "1 dozen Bone-in Wings, hot sauce",
        "1 large Coke",
        "1 large Pepperoni",
    ]


@pytest.mark.parametrize(
    "items, mode, fulfillment, expected",
    [
        (
            [item("pepperoni", "medium"), item("wings", "dozen", sauce="hot")],
            "pickup",
            {},
            "37.04",
        ),
        (
            [item("cheese", "medium")],
            "delivery",
            {"address": "300 De Haro St", "zip_code": "94107"},
            "24.32",
        ),
        (
            [item("pepperoni", "large")],
            "delivery",
            {"address": "123 Market St", "zip_code": "94105"},
            "26.78",
        ),
        (
            [item("cheese", "large"), item("wings", "dozen", sauce="bbq")],
            "pickup",
            {},
            "38.13",
        ),
        ([item("margherita", "medium")], "pickup", {}, "17.43"),
        (
            [
                item("pepperoni", "large", 2),
                item("meat_lovers", "large"),
                item("wings", "dozen", sauce="hot"),
                item("coke", "large", 2),
            ],
            "pickup",
            {},
            "94.76",
        ),
        (
            [
                item(
                    "cheese",
                    "large",
                    first_half=["mushrooms"],
                    second_half=["pepperoni"],
                ),
                item("lemonade", "large"),
            ],
            "pickup",
            {},
            "26.14",
        ),
        ([item("pepperoni", "large")], "pickup", {}, "21.79"),
        (
            [item("cheese", "medium"), item("wings", "dozen", sauce="lemon_pepper")],
            "pickup",
            {},
            "34.86",
        ),
        ([item("hawaiian", "large")], "pickup", {}, "22.88"),
        (
            [item("boneless", "half_dozen", sauce="plain"), item("caesar")],
            "pickup",
            {},
            "19.05",
        ),
    ],
)
async def test_scenario_totals(items, mode, fulfillment, expected):
    pos = await order(*items, mode=mode, **fulfillment)
    assert total(pos) == Decimal(expected)


async def test_a_small_order_pays_the_small_order_fee_and_a_tip_goes_on_top():
    pos = await order(item("cheese", "medium"))
    assert pos.quote().small_order_fee == Decimal("2.99")
    await pos.set_tip(dollars=5)
    big = await order(item("pepperoni", "large"))
    await big.set_tip(dollars=5)
    assert total(big) == Decimal("26.79")
    await big.set_tip(percent=10)
    assert big.tip == Decimal("2.00")


async def test_what_cant_go_on_comes_back_with_what_to_ask_and_the_rest_goes_on():
    pos = Pizzeria()
    added, problems = await pos.add_items(
        [
            item("pepperoni", "large"),
            item("coke"),
            item("wings", "dozen"),
            item("buffalo_cauliflower"),
            item("fries", toppings=["bacon"]),
            item("veggie", "large", without=["ham"]),
        ]
    )
    assert [ln.label for ln in added] == ["large Pepperoni", "dozen Bone-in Wings"]
    assert any("Ask which size Coke" in p for p in problems)
    assert any("sold out" in p and "Mozzarella Sticks" in p for p in problems)
    assert any("Only pizzas" in p for p in problems)
    assert any("doesn't come with ham" in p for p in problems)
    assert [ln.id for ln in pos.lines] == ["L1", "L2"]


async def test_saying_an_item_again_never_adds_it_twice():
    pos = Pizzeria()
    await pos.add_items([item("pepperoni", "large", 2)])
    added, problems = await pos.add_items([item("pepperoni", "large", 2)])
    assert not added and "already on the order as L1" in problems[0]
    assert [ln.quantity for ln in pos.lines] == [2]


async def test_wings_without_a_sauce_go_on_and_wait_for_the_callers_pick():
    sent = []

    async def publish(parts):
        sent.append(parts)

    pos = Pizzeria(publish)
    _, problems = await pos.add_items(
        [item("pepperoni", "large"), item("wings", "dozen")]
    )
    assert not problems
    await pos.set_fulfillment("pickup")
    await pos.set_contact(**SAM)
    assert pos.cart_facts()["order"][1]["needs"] == "sauce"
    assert sent[-1]["cart"]["lines"][1]["needs"] == "sauce"
    with pytest.raises(PizzeriaError, match="Which sauce on L2"):
        pos.order_read_back()
    await pos.update_item("L2", sauce="lemon_pepper")
    assert "stillToAsk" not in pos.cart_facts()
    assert pos.order_read_back()["items"] == [
        "1 large Pepperoni",
        "1 dozen Bone-in Wings, lemon pepper sauce",
    ]


async def test_naming_the_sauce_again_is_for_the_waiting_wings_not_another_order():
    pos = Pizzeria()
    await pos.add_items([item("wings", "dozen")])
    added, problems = await pos.add_items([item("wings", "dozen", sauce="hot")])
    assert not added and "update_item L1" in problems[0]
    added, _ = await pos.add_items([item("boneless", "dozen", sauce="hot")])
    assert [ln.label for ln in added] == ["dozen Boneless Wings"]
    # two orders asked for at once are two orders, even while one waits on its sauce
    both = Pizzeria()
    added, problems = await both.add_items(
        [item("wings", "dozen"), item("wings", "dozen", sauce="hot")]
    )
    assert len(added) == 2 and not problems


async def test_plain_wings_have_no_sauce_on_them():
    pos = await order(item("boneless", "half_dozen", sauce="plain"))
    assert pos.order_read_back()["items"] == ["1 half dozen Boneless Wings, plain"]


async def test_an_order_thats_not_placed_is_dropped_whole_and_only_before_its_placed():
    pos = await order(item("margherita", "medium"), item("coke", "large"))
    await pos.set_tip(dollars=3)
    assert pos.drop_read_back() == {
        "drop": ["1 medium Margherita", "1 large Coke"],
        "placed": False,
        "due": "$0.00",
    }
    await pos.drop()
    assert pos.lines == [] and pos.tip == 0 and pos.quote() is None
    with pytest.raises(PizzeriaError, match="nothing to drop"):
        pos.drop_read_back()
    placed = await order(item("margherita", "medium"))
    await placed.place()
    with pytest.raises(PizzeriaError, match="already sent"):
        placed.drop_read_back()


async def test_an_update_changes_only_what_it_names():
    pos = Pizzeria()
    await pos.add_items(
        [item("margherita", "small", toppings=["mushrooms"]), item("caesar")]
    )
    line, changed = await pos.update_item("L1", size="large")
    assert changed and line.describe() == "1 large Margherita, add mushrooms"
    line, changed = await pos.update_item("L1", toppings=[])
    assert changed and line.describe() == "1 large Margherita"
    before = pos.revision
    line, changed = await pos.update_item("L1", size="large")
    assert not changed and pos.revision == before
    assert pos.lines[1].describe() == "1 Caesar Salad"


async def test_making_the_coke_a_sprite_keeps_its_size_and_count():
    pos = Pizzeria()
    await pos.add_items([item("coke", "large", 2)])
    line, _ = await pos.update_item("L1", item_id="sprite")
    assert line.describe() == "2 large Sprite"


async def test_a_half_and_half_is_one_pizza_at_half_a_topping_each_side():
    pos = Pizzeria()
    _, problems = await pos.add_items(
        [item("cheese", "large", first_half=["mushrooms"], second_half=["mushrooms"])]
    )
    assert "whole pizza" in problems[0]
    added, _ = await pos.add_items(
        [item("cheese", "large", first_half=["mushrooms"], second_half=["pepperoni"])]
    )
    assert added[0].unit_price == Decimal("19.99")
    assert added[0].modifiers == ["half mushrooms", "other half pepperoni"]


async def test_pickup_has_no_driver_and_delivery_keeps_what_isnt_restated():
    pos = await order(item("pepperoni", "large"))
    with pytest.raises(PizzeriaError, match="pickup order"):
        await pos.set_fulfillment(driver_note="leave it at the door")
    await pos.set_fulfillment("delivery", address="123 Market St", zip_code="94105")
    await pos.set_fulfillment(driver_note="Leave it at the door, don't ring the bell")
    f = pos.fulfillment
    assert (f.address, f.zip_code) == ("123 Market St", "94105")
    with pytest.raises(PizzeriaError, match="street address"):
        await Pizzeria().set_fulfillment("delivery")


@pytest.mark.parametrize(
    "phone", ["415555014", "0415550142", "4155550142 1", "9115550142"]
)
async def test_a_callback_number_must_be_a_real_ten_digit_number(phone):
    with pytest.raises(PizzeriaError, match="ten-digit"):
        await Pizzeria().set_contact(phone=phone)


async def test_an_order_is_placed_only_once_it_has_everything():
    pos = Pizzeria()
    with pytest.raises(PizzeriaError, match="Nothing"):
        pos.order_read_back()
    await pos.add_items([item("pepperoni", "large")])
    with pytest.raises(PizzeriaError, match="set_pickup or set_delivery"):
        await pos.place()
    await pos.set_fulfillment("pickup")
    with pytest.raises(PizzeriaError, match="no name"):
        await pos.place()
    await pos.set_contact(**SAM)
    read_back = pos.order_read_back()
    assert read_back["total"] == "$21.79" and read_back["payment"].startswith(
        "Pay at pickup"
    )
    placed = await pos.place()
    assert placed.number == "215" and "20 minutes" in placed.ready_in
    with pytest.raises(PizzeriaError, match="already sent"):
        await pos.add_items([item("coke", "large")])


async def test_cancelling_needs_a_placed_order_and_leaves_nothing_due():
    pos = await order(item("margherita", "medium"))
    with pytest.raises(PizzeriaError, match="Nothing has been placed"):
        pos.cancel_read_back()
    await pos.place()
    assert pos.cancel_read_back()["due"] == "$0.00"
    await pos.cancel()
    assert pos.order.cancelled and pos.due_now() == 0


async def test_the_allergen_lookup_names_items_sauces_and_toppings():
    pos = Pizzeria()
    contains = await pos.check_allergen("nuts", severe=False)
    assert contains == ["Pesto Chicken", "wings with Thai chili peanut sauce"]
    assert "extra cheese as a topping" in await pos.check_allergen("dairy", severe=True)
    assert pos.allergies == {"nuts": False, "dairy": True}


# ---- the order on its way ----


async def on_its_way():
    return Pizzeria.out_for_delivery("Sam Carter")


async def test_order_214_is_due_at_the_door_and_the_sprite_ran_out():
    pos = await on_its_way()
    assert pos.delivery.due == Decimal("49.65")
    assert pos.sold_out_unresolved()
    assert pos.same_price_swaps() == ["large Coke", "large Diet Coke", "large Lemonade"]


async def test_swapping_and_taking_off_the_sprite_and_changing_their_mind():
    pos = await on_its_way()
    assert await pos.take_off_sold_out_item() == Decimal("4.35")
    assert pos.balance() == Decimal("-4.35")
    await pos.swap_sold_out_item("coke")
    assert pos.balance() == 0 and not pos.sold_out_unresolved()
    assert pos.changes() == ["large Sprite swapped for a large Coke, same price"]
    with pytest.raises(PizzeriaError, match="same price"):
        await pos.swap_sold_out_item("tiramisu")


async def test_breadsticks_are_set_as_a_total_and_only_two_fit():
    pos = await on_its_way()
    await pos.swap_sold_out_item("coke")
    assert await pos.set_breadsticks(1) == Decimal("6.53")
    await pos.set_breadsticks(2)
    assert pos.due_now() == Decimal("62.71") and pos.balance() == Decimal("13.06")
    with pytest.raises(PizzeriaError, match="at most 2"):
        await pos.set_breadsticks(3)


async def test_the_read_back_needs_the_sprite_settled_and_cancelling_clears_whats_due():
    pos = await on_its_way()
    with pytest.raises(PizzeriaError, match="still on the order"):
        await pos.read_back_changes()
    await pos.take_off_sold_out_item()
    await pos.set_breadsticks(1)
    await pos.set_driver_note("Leave it at the door")
    facts = await pos.read_back_changes()
    assert facts["dueAtTheDoor"] == "$51.83"
    assert facts["comparedToBefore"] == "$2.18 more than before"
    other = await on_its_way()
    await other.cancel()
    assert other.balance() == Decimal("-49.65")
    with pytest.raises(PizzeriaError, match="cancelled"):
        await other.set_tip(dollars=5)


async def test_a_wrong_number_changes_nothing_and_shares_nothing():
    pos = await on_its_way()
    await pos.end_early("wrong_person")
    with pytest.raises(PizzeriaError, match="isn't the customer"):
        await pos.swap_sold_out_item("coke")
    assert grade(pos, {"ended_early": "wrong_person", "unchanged": True}) == []


async def test_the_grader_fails_a_run_that_ends_in_the_wrong_state():
    pos = await order(item("pepperoni", "large"))
    assert grade(pos, {"placed": True})[0].startswith("never placed")
    await pos.place()
    assert (
        grade(pos, {"placed": True, "cart": ["1 large Pepperoni"], "total": "21.79"})
        == []
    )
    assert grade(pos, {"total": "44.66"}) == ["total 21.79, expected 44.66"]


async def test_the_page_gets_each_change_in_one_update():
    updates = []

    async def publish(parts):
        updates.append(parts)

    pos = Pizzeria.out_for_delivery(None, publish)
    await pos.publish_all()
    await pos.swap_sold_out_item("coke")
    assert len(updates) == 2
    assert set(updates[1]) == {"cart", "fulfillment", "quote", "delivery"}
    sent = updates[0] | updates[1]
    assert set(sent) == {
        "menu",
        "allergies",
        "order",
        "cart",
        "fulfillment",
        "quote",
        "delivery",
    }
    wings, coke = sent["cart"]["lines"][1:]
    assert wings["size"] == "12" and wings["modifiers"] == ["hot sauce"]
    assert coke["was"] == "Large Sprite" and not coke["out"]
    assert sent["delivery"]["dueNow"] == 49.65
    assert sent["delivery"]["changes"] == [
        "large Sprite swapped for a large Coke, same price"
    ]
    await pos.cancel()
    assert (
        updates[2]["delivery"]["cancelled"] and updates[2]["delivery"]["changes"] == []
    )
