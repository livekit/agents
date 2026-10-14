"""Grades a simulated call on what the POS ended up with."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from backend import DELIVERY_FEE, TAX_RATE, Pos


def grade(pos: Pos, expect: dict[str, Any]) -> list[str]:
    """The problems with the POS's final state, given a scenario's `expect`.

    Keys (all optional):
        submitted: whether an order must (true) or mustn't (false) reach the kitchen.
        cart: the exact submitted lines, e.g. ["1 × large Pepperoni"], in any order.
        cart_includes: lines that must be in the submitted cart, the rest is the caller's pick.
        not_in_cart: menu ids that mustn't be in the cart at the end.
        fulfillment: "pickup" or "delivery".
        zip: the delivery ZIP.
        total: the submitted total, e.g. "44.66".
        allergy: the allergen, or list of allergens, items_with_allergen must have been asked
            about, e.g. "nuts" or ["nuts", "dairy"].
        name, phone: the contact on the order, phone as "(415) 555-0142".

    The delivery-update call (outbound) also takes:
        balance: owed (+) or refunded (-) against the original charge, e.g. "-4.35".
        driver_note: words the driver note must contain, case-insensitive, e.g. ["4512"].
        staff_note: whether a note for the manager must (true) or mustn't (false) exist.
        cancel_refused: whether the customer asked to cancel (and was refused).
        ended_early: why the call ended without the update: wrong_person, call_back_later, or
            declined. Leave it out when the update must be finished.
        unchanged: true when nothing on the order may change on the call.
    Every outbound call must end with the out-of-stock item resolved and read back, unless
    it ended early.
    """
    problems: list[str] = []
    lines = sorted(ln.describe().split(": ", 1)[1] for ln in pos.lines)
    order = pos.order

    if "submitted" in expect:
        if expect["submitted"] and order is None:
            problems.append(f"no order was submitted (cart: {lines or 'empty'})")
        elif not expect["submitted"] and order is not None:
            problems.append(f"order #{order.number} was submitted but shouldn't have been")
    if order is not None:
        # whatever the scenario, a submitted order is priced right
        q = pos.quote
        fee = DELIVERY_FEE if pos.fulfillment.mode == "delivery" else Decimal("0")
        tax = (pos.subtotal() * TAX_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if q is None or not pos.quote_current:
            problems.append("submitted at a total that wasn't current")
        elif (q.subtotal, q.tax, q.delivery_fee, q.total) != (
            pos.subtotal(),
            tax,
            fee,
            pos.subtotal() + tax + fee,
        ):
            problems.append(f"total ${q.total} doesn't add up")
    if (cart := expect.get("cart")) is not None and sorted(cart) != lines:
        problems.append(f"cart is {lines}, expected {sorted(cart)}")
    for want in expect.get("cart_includes", []):
        if want not in lines:
            problems.append(f"cart is missing {want!r} (cart: {lines})")
    for item_id in expect.get("not_in_cart", []):
        if any(ln.item.id == item_id for ln in pos.lines):
            problems.append(f"{item_id} is in the cart")
    if (mode := expect.get("fulfillment")) and pos.fulfillment.mode != mode:
        problems.append(f"fulfillment is {pos.fulfillment.mode}, expected {mode}")
    if (zip_code := expect.get("zip")) and pos.fulfillment.zip != zip_code:
        problems.append(f"delivery ZIP is {pos.fulfillment.zip}, expected {zip_code}")
    if (total := expect.get("total")) is not None:
        got = pos.quote.total if pos.quote else None
        if got != Decimal(str(total)):
            problems.append(f"total is ${got}, expected ${total}")
    allergy = expect.get("allergy") or []
    for allergen in [allergy] if isinstance(allergy, str) else allergy:
        if allergen not in pos.allergies:
            problems.append(f"the {allergen} allergy was never looked up in the menu data")
    for key in ("name", "phone"):
        if (want := expect.get(key)) and getattr(pos, key) != want:
            problems.append(f"{key} is {getattr(pos, key)!r}, expected {want!r}")
    if (d := pos.delivery) is not None:
        if d.ended_early != expect.get("ended_early"):
            problems.append(
                f"the call ended early as {d.ended_early}, expected {expect.get('ended_early')}"
            )
        if d.ended_early is None:
            if out := pos.out_of_stock():
                problems.append(f"{out[0].item.name} is out of stock and still on the order")
            if not pos.read_back_current:
                problems.append("the final changes were never read back")
        if expect.get("unchanged") and d.changes:
            problems.append(f"the order changed: {'; '.join(d.changes.values())}")
        if (bal := expect.get("balance")) is not None and pos.balance() != Decimal(str(bal)):
            problems.append(f"balance is ${pos.balance()}, expected ${bal}")
        for word in expect.get("driver_note", []):
            if word.lower() not in (d.note or "").lower():
                problems.append(f"driver note {d.note!r} is missing {word!r}")
        if "staff_note" in expect and bool(d.staff_note) != expect["staff_note"]:
            problems.append(f"manager note is {d.staff_note!r}, expected {expect['staff_note']}")
        if "cancel_refused" in expect and d.cancel_refused != expect["cancel_refused"]:
            problems.append(
                f"cancel refused is {d.cancel_refused}, expected {expect['cancel_refused']}"
            )
    return problems
