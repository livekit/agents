"""Grades a simulated call on what the point of sale ended up with, against the scenario's
`expected`.

Keys for a call in, all optional:
    placed: true if the order must reach the kitchen, false if it mustn't
    cart: the exact lines at the end, e.g. ["1 large Pepperoni", "1 dozen Bone-in Wings, hot sauce"]
    cart_includes: lines that must be among them, the rest being the caller's pick
    not_in_cart: menu ids that mustn't be on the order
    fulfillment: pickup or delivery
    zip: the delivery ZIP code
    total: what the order comes to, e.g. "44.66"
    tip: the tip in that total, e.g. "5.00"
    name, phone: the contact on the order, phone as "(415) 555-0142"
    driver_note: what the note for the driver must mention, e.g. ["door", "bell"]
    allergies: the allergens that must have been looked up, e.g. [nuts]
    cancelled: true if the order must end up cancelled, false if it mustn't
Keys for the delivery-update call, also optional:
    cart, not_in_cart, total, tip, driver_note, cancelled: as above
    balance: how much more (+) or less (-) is due at the door than when it was placed, e.g. "-4.35"
    ended_early: wrong_person, call_back_later, or declined; leave it out when the call must go
        through the update
    unchanged: true if nothing on the order may change
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from livekit.agents import SimulationContext

from pizzeria import Pizzeria


async def on_simulation_end(sim: SimulationContext) -> None:
    expected = sim.userdata().get("expected")
    if not expected:
        return
    pos: Pizzeria = sim.job_context.primary_session.userdata
    if problems := grade(pos, expected):
        sim.fail(reason="; ".join(problems))


def grade(pos: Pizzeria, expected: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    lines = sorted(ln.describe() for ln in pos.lines)
    order = pos.order
    cancelled = bool(order and order.cancelled)

    if "placed" in expected and (order is not None) != expected["placed"]:
        problems.append(
            f"order #{order.number} was placed"
            if order
            else f"never placed (order: {lines})"
        )
    if (cart := expected.get("cart")) is not None and sorted(cart) != lines:
        problems.append(f"order is {lines}, expected {sorted(cart)}")
    for want in expected.get("cart_includes", []):
        if want not in lines:
            problems.append(f"{want!r} isn't on the order ({lines})")
    for item_id in expected.get("not_in_cart", []):
        if any(ln.item.id == item_id for ln in pos.lines):
            problems.append(f"{item_id} is on the order")
    if (mode := expected.get("fulfillment")) and pos.fulfillment.mode != mode:
        problems.append(f"{pos.fulfillment.mode}, expected {mode}")
    if (zip_code := expected.get("zip")) and pos.fulfillment.zip_code != zip_code:
        problems.append(f"ZIP {pos.fulfillment.zip_code}, expected {zip_code}")
    if (total := expected.get("total")) is not None:
        quote = pos.quote()
        if quote is None or quote.total != Decimal(total):
            problems.append(f"total {quote.total if quote else None}, expected {total}")
    if (tip := expected.get("tip")) is not None and pos.tip != Decimal(tip):
        problems.append(f"tip {pos.tip}, expected {tip}")
    for key in ("name", "phone"):
        if (want := expected.get(key)) and getattr(pos, key) != want:
            problems.append(f"{key} {getattr(pos, key)!r}, expected {want!r}")
    note = (pos.fulfillment.driver_note or "").lower()
    for word in expected.get("driver_note", []):
        if word.lower() not in note:
            problems.append(f"the driver note {note!r} doesn't mention {word!r}")
    for allergen in expected.get("allergies", []):
        if allergen not in pos.allergies:
            problems.append(f"never looked up {allergen}")
    if "cancelled" in expected and cancelled != expected["cancelled"]:
        problems.append(
            "cancelled the order" if cancelled else "never cancelled the order"
        )

    if (d := pos.delivery) is not None:
        if d.ended_early != expected.get("ended_early"):
            problems.append(
                f"ended early as {d.ended_early}, expected {expected.get('ended_early')}"
            )
        if d.ended_early is None and not cancelled:
            if pos.sold_out_unresolved():
                problems.append(f"the {pos.ran_out_line().label} is still on the order")
            if d.read_back_revision != pos.revision:
                problems.append("the changes were never read back")
        if expected.get("unchanged") and pos.changes():
            problems.append(f"the order changed: {pos.changes()}")
        if (
            balance := expected.get("balance")
        ) is not None and pos.balance() != Decimal(balance):
            problems.append(f"balance {pos.balance()}, expected {balance}")
    return problems
