"""Grades a simulated call on what the POS ended up with."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from backend import DELIVERY_FEE, SMALL_ORDER_FEE, SMALL_ORDER_UNDER, TAX_RATE, Pos


def grade(pos: Pos, expect: dict[str, Any]) -> list[str]:
    """The problems with the POS's final state, given a scenario's `expect`.

    Keys (all optional):
        submitted: whether an order must (true) or mustn't (false) reach the kitchen.
        cart: the exact submitted lines, e.g. ["1 × large Pepperoni"], in any order.
        cart_includes: lines that must be in the submitted cart, the rest is the caller's pick.
        not_in_cart: menu ids that mustn't be in the cart at the end.
        fulfillment: "pickup" or "delivery".
        zip: the delivery ZIP.
        driver_note: words the drop-off note for the driver must contain, case-insensitive,
            e.g. ["door"]; on the delivery-update call, the note saved on that call.
        total: the submitted total, e.g. "44.66".
        tip: the tip in that total, e.g. "5.00" ("0" for none).
        cancelled: whether the order placed on the call must (true) or mustn't (false) end up
            cancelled.
        allergy: the allergen, or list of allergens, items_with_allergen must have been asked
            about, e.g. "nuts" or ["nuts", "dairy"].
        name, phone: the contact on the order, phone as "(415) 555-0142".

    The delivery-update call (outbound) also takes:
        balance: how much more (+) or less (-) is due at the door than when the order went
            out, e.g. "-4.35"; a cancelled order is its whole original amount less.
        ended_early: why the call ended without the update: wrong_person, call_back_later, or
            declined. Leave it out when the update must be finished.
        unchanged: true when nothing on the order may change on the call.
    Every outbound call must end with the out-of-stock item resolved (or the order cancelled)
    and read back, unless it ended early.
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
        small = SMALL_ORDER_FEE if pos.subtotal() < SMALL_ORDER_UNDER else Decimal("0")
        tax = (pos.subtotal() * TAX_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if q is None or not pos.quote_current:
            problems.append("submitted at a total that wasn't current")
        elif (q.subtotal, q.tax, q.delivery_fee, q.small_order_fee, q.total) != (
            pos.subtotal(),
            tax,
            fee,
            small,
            pos.subtotal() + tax + fee + small + q.tip,
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
    if (tip := expect.get("tip")) is not None and pos.tip != Decimal(str(tip)):
        problems.append(f"tip is ${pos.tip}, expected ${tip}")
    if "cancelled" in expect:
        cancelled = bool(order and order.cancelled)
        if cancelled != expect["cancelled"]:
            problems.append(f"cancelled is {cancelled}, expected {expect['cancelled']}")
    note = pos.delivery.note if pos.delivery else pos.fulfillment.note
    for word in expect.get("driver_note", []):
        if word.lower() not in (note or "").lower():
            problems.append(f"driver note {note!r} is missing {word!r}")
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
            if (out := pos.out_of_stock()) and not (order and order.cancelled):
                problems.append(f"{out[0].item.name} is out of stock and still on the order")
            if not pos.read_back_current:
                problems.append("the final changes were never read back")
        if expect.get("unchanged") and d.changes:
            problems.append(f"the order changed: {'; '.join(d.changes.values())}")
        if (bal := expect.get("balance")) is not None and pos.balance() != Decimal(str(bal)):
            problems.append(f"balance is ${pos.balance()}, expected ${bal}")
    return problems
