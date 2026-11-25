"""The point of sale on its own: the restaurant's rules, prices, and what the page is sent."""

from decimal import Decimal

import pytest

from pizzeria import Pizzeria, PizzeriaError

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


async def on_its_way():
    return Pizzeria.out_for_delivery("Sam Carter")


async def test_swapping_and_taking_off_the_sprite_and_changing_their_mind():
    pos = await on_its_way()
    assert await pos.take_off_sold_out_item() == Decimal("4.35")
    assert pos.balance() == Decimal("-4.35")
    await pos.swap_sold_out_item("coke")
    assert pos.balance() == 0 and not pos.sold_out_unresolved()
    assert pos.changes() == ["large Sprite swapped for a large Coke, same price"]
    with pytest.raises(PizzeriaError, match="same price"):
        await pos.swap_sold_out_item("tiramisu")


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
