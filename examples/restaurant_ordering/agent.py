"""Zappi's Pizzeria on the phone: Rosa takes pickup and delivery orders, and calls a customer
back when an item in their delivery runs out.

Calls are inbound by default. `{"direction": "outbound"}` in the job metadata (or
CALL_DIRECTION=outbound for `console`) places the delivery-update call instead: to a real phone
when the metadata also has `"dial"` and SIP_OUTBOUND_TRUNK_ID is set, otherwise to whoever is in
the room, who plays the customer.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass

from backend import (
    DELIVERY_FEE,
    DELIVERY_MINIMUM,
    ENDED_EARLY,
    MAX_EXTRAS,
    SAUCES,
    TAX_RATE,
    TOPPINGS,
    EndedEarly,
    Line,
    Pos,
    PosError,
    demo_pos,
)
from call import (
    VOICE_RULES,
    end_after_reply,
    first_name,
    hang_up,
    pause_between_sentences,
    room_options,
    spoken_number,
    transfer,
)
from dotenv import load_dotenv
from grade import grade

from livekit import api
from livekit.agents import (
    AMD,
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    SimulationContext,
    ToolError,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
    room_io,
)

logger = logging.getLogger("restaurant-ordering")

load_dotenv()

PERSONA = "Rosa"
BUSINESS = "Zappi's Pizzeria"
VOICE = "f786b574-daa5-4673-aa0c-cbe3e8534c02"  # Cartesia "Katie"
# the number a customer calls back on, said on voicemail
RESTAURANT_NUMBER = os.getenv("RESTAURANT_NUMBER", "")
# the participant identity of a phone the agent rings
PHONE = "phone"


@dataclass
class Userdata:
    pos: Pos
    transferred: bool = False  # handed to a staff member: the call is theirs now


Ctx = RunContext[Userdata]


@contextlib.contextmanager
def _pos_rules(ctx: Ctx) -> Iterator[Pos]:
    """Run a POS call; a refusal goes back to the agent, as a tool error, to relay."""
    try:
        yield ctx.userdata.pos
    except PosError as e:
        raise ToolError(str(e)) from e


def _line_note(pos: Pos, line: Line) -> str:
    # every allergy the caller mentioned stays in play for everything added after it
    if hits := sorted(line.allergens.intersection(pos.allergies)):
        return (
            f" Heads up: per the menu data this contains {' and '.join(hits)}, which the "
            "caller mentioned an allergy to. Tell them."
        )
    return ""


_MORE = re.compile(r"\b(more|another|add|extra|additional)\b")
_NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty"
).split()


def _asked_for_more(ctx: Ctx, total: int) -> bool:
    # whether the caller's last words ask for a bigger count ("two more", "another
    # one") or state the new total ("make it three")
    said = next(
        (
            m.text_content or ""
            for m in reversed(ctx.session.history.messages())
            if m.role == "user"
        ),
        "",
    ).lower()
    total_words = {str(total)} | ({_NUMBERS[total]} if total < len(_NUMBERS) else set())
    return bool(_MORE.search(said)) or any(re.search(rf"\b{w}\b", said) for w in total_words)


class RosaAgent(Agent):
    def __init__(self, pos: Pos) -> None:
        super().__init__(instructions=_instructions(pos))
        self._submitted_at: float | None = None
        self._warned = False

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions=(
                f'Answer the phone: "Thanks for calling {BUSINESS}, this is {PERSONA}. '
                'What can I get started for you?"'
            )
        )

    # ---- the menu ----

    @function_tool
    async def get_menu(self, ctx: Ctx) -> str:
        """The full menu with what's available right now. Call it when the caller asks to hear
        the whole menu (then read every available item, category by category), or if you're
        unsure an item exists or is available."""
        pos = ctx.userdata.pos
        return "\n".join(
            m.describe() + ("" if m.available else " Sold out today: not available.")
            for m in pos.menu
        )

    @function_tool
    async def check_availability(self, ctx: Ctx, item_id: str) -> str:
        """Check that an item can be ordered today.

        Args:
            item_id: The menu id, e.g. "buffalo_cauliflower".
        """
        with _pos_rules(ctx) as pos:
            item = pos.check_availability(item_id)
        return f"{item.name} is available."

    @function_tool
    async def items_with_allergen(self, ctx: Ctx, allergen: str, severe: bool = False) -> str:
        """Look up which menu items contain an allergen, from the menu data. Call this for any
        allergy or "gluten-free" question before answering; never answer from memory. One
        allergen per call: for two, call it twice.

        Args:
            allergen: What the caller is allergic to, in their words, e.g. "nuts",
                "gluten-free", "lactose intolerant".
            severe: True only if they said it's severe (anaphylaxis, an EpiPen, "really
                bad"). Don't assume it from the allergen alone.
        """
        with _pos_rules(ctx) as pos:
            tag, items = pos.with_allergen(allergen)
            pos.note_allergy(tag, severe)
        found = (
            f"Per the menu data, these contain {tag}: {', '.join(items)}."
            if items
            else f"No item on the menu lists {tag}."
        )
        clear = [m.name for m in pos.menu if m.available and tag not in m.allergens]
        out = (
            f"{found} These don't list {tag}: {', '.join(clear) or 'nothing'}. But it's all made "
            f"in one shared kitchen, so you can't promise anything is free of {tag}. Never say an "
            f'item is "safe" or "{tag}-free". Answer what they asked in a sentence or two; when a '
            'list is long, sum it up ("all the pizzas and the wings") instead of reading out '
            "every item."
        )
        if severe:
            out += " It's a severe allergy: offer to transfer them to a staff member."
        return out

    # ---- the cart ----

    @function_tool
    async def add_item(
        self,
        ctx: Ctx,
        item_id: str,
        quantity: int = 1,
        size: str | None = None,
        toppings: list[str] | None = None,
        sauce: str | None = None,
        without: list[str] | None = None,
    ) -> str:
        """Add a new item to the cart. Don't call it for an item that's already there: a
        caller naming it again is repeating themselves, not ordering more.

        Args:
            item_id: The menu id, e.g. "pepperoni".
            quantity: How many orders of it: "a dozen wings" is quantity 1, size dozen.
            size: Required for pizza (small, medium, large), wings (half dozen, dozen), and
                drinks with sizes (e.g. can, 2-liter). Leave out for one-size items.
            toppings: Extra pizza toppings, e.g. ["mushrooms"]. Pizza only.
            sauce: Required for wings, e.g. "hot" or "Thai".
            without: What to leave off a pizza ("no olives"), e.g. ["olives"]. Same price.
        """
        with _pos_rules(ctx) as pos:
            line = pos.add(item_id, quantity, size, toppings, sauce, without)
        return f"Added {line.describe()}.{_line_note(pos, line)} {pos.cart_summary()}"

    @function_tool
    async def update_item(
        self,
        ctx: Ctx,
        line_id: str,
        quantity: int | None = None,
        size: str | None = None,
        toppings: list[str] | None = None,
        sauce: str | None = None,
        without: list[str] | None = None,
    ) -> str:
        """Change one line in the cart. Pass only what changed; the rest stays as it is.

        Args:
            line_id: The cart line, e.g. "L1".
            quantity: The new total for this line, not how many to add. "Two pepperonis"
                when the line already has two is a repeat: no change.
            size: The new size.
            toppings: The full new list of extra toppings for this pizza.
            sauce: The new wing sauce.
            without: The full new list of what to leave off this pizza, e.g. ["olives"]; []
                puts everything back.
        """
        more = quantity is not None and _asked_for_more(ctx, quantity)
        with _pos_rules(ctx) as pos:
            before = pos.line(line_id).describe()
            line = pos.update(line_id, quantity, size, toppings, sauce, without, more=more)
        after = line.describe()
        if after == before:
            return f"No change: it's already {after}."
        return f"Updated {after}.{_line_note(pos, line)} {pos.cart_summary()}"

    @function_tool
    async def remove_item(self, ctx: Ctx, line_id: str) -> str:
        """Take one line out of the cart.

        Args:
            line_id: The cart line, e.g. "L3".
        """
        with _pos_rules(ctx) as pos:
            line = pos.remove(line_id)
        return f"Removed {line.describe()}. {pos.cart_summary()}"

    @function_tool
    async def offer_breadsticks(self, ctx: Ctx) -> str:
        """Call this right before suggesting breadsticks, the only upsell. It refuses if
        they were already offered or ordered, and then you mustn't suggest them."""
        with _pos_rules(ctx) as pos:
            item = pos.offer_upsell()
        return (
            f"Offer {item.name.lower()} ({item.blurb.lower()}, ${item.sizes[0].price}) in one "
            "short sentence. If they say no, drop it."
        )

    # ---- pickup or delivery, and who it's for ----

    @function_tool
    async def set_pickup(self, ctx: Ctx) -> str:
        """The caller will pick the order up."""
        with _pos_rules(ctx) as pos:
            pos.set_pickup()
        return "Pickup it is."

    @function_tool
    async def set_delivery(self, ctx: Ctx, address: str) -> str:
        """Set the order to delivery, to any full street address.

        Args:
            address: The street address as the caller said it, e.g. "300 De Haro St, 94107".
        """
        with _pos_rules(ctx) as pos:
            pos.set_delivery(address)
        out = f"Delivery to {address}. The delivery fee is ${DELIVERY_FEE}."
        if short := pos.below_minimum():
            out += (
                f" But delivery needs a ${DELIVERY_MINIMUM} subtotal and it's ${pos.subtotal()} "
                f"now, ${short} short. Tell them, and offer to add something or switch to pickup."
            )
        # the cart as it stands, so anything the caller asked for but isn't in it gets added
        return f"{out} {pos.cart_summary()}"

    @function_tool
    async def set_contact(self, ctx: Ctx, name: str | None = None, phone: str | None = None) -> str:
        """Save the caller's name and callback number for the order. Pass either or both, and
        only what the caller actually said: never a placeholder.

        Args:
            name: The name for the order.
            phone: The callback number however they said it; you can pass just the digits.
                Only what they said: never fill in, pad, or guess a digit.
        """
        if not (name or "").strip() and not (phone or "").strip():
            raise ToolError("The caller hasn't given a name or number yet. Ask for both first.")
        with _pos_rules(ctx) as pos:
            pos.set_contact(name, phone)
        return f"Saved: name {pos.name or 'missing'}, phone {pos.phone or 'missing'}."

    # ---- read-back and submit ----

    @function_tool
    async def price_order(self, ctx: Ctx) -> str:
        """Work out the total for the cart as it stands, to read back to the caller. Call it
        again after any change: an order can only be submitted at a total that's current."""
        with _pos_rules(ctx) as pos:
            q = pos.price()
        out = f"{pos.cart_summary()} {q.describe()} (tax is {TAX_RATE:.0%} of the subtotal)."
        if short := pos.below_minimum():
            out += f" Delivery is ${short} under the ${DELIVERY_MINIMUM} minimum."
        return out

    @function_tool
    async def submit_order(self, ctx: Ctx) -> str:
        """Send the order to the kitchen. Only after you've read back the order and total
        from price_order and the caller said it's right."""
        with _pos_rules(ctx) as pos:
            order = pos.submit()
        self._submitted_at = time.time()
        return (
            f"Order number {order.number} is in, {order.ready_in}. Now tell the caller the "
            "order number and when it'll be ready, and ask if there's anything else. Don't "
            "end the call yet."
        )

    @function_tool
    async def transfer_to_staff(self, ctx: Ctx, reason: str) -> str:
        """Hand the call to a staff member at the restaurant. Offer it for a severe allergy, or
        if the caller asks for a person; transfer only after they say yes.

        Args:
            reason: Why, in a few words.
        """
        ctx.userdata.transferred = True
        await transfer(ctx, "someone at the restaurant", reason)
        return "Transferred to restaurant staff."

    @function_tool
    async def end_call(self, ctx: Ctx) -> None:
        """Hang up, once you've said goodbye and the caller has nothing else."""
        ud = ctx.userdata
        await ctx.wait_for_playout()
        order = ud.pos.order
        spoken = [
            m.created_at
            for m in ctx.session.history.messages()
            if m.role == "assistant" and m.text_content
        ]
        # a submitted order isn't done until the caller has heard its number and pickup time
        if order and not any(t > (self._submitted_at or 0) for t in spoken):
            raise ToolError(
                f"Tell the caller their order number, {order.number}, and that it'll be "
                f"{order.ready_in}, then say goodbye before ending the call."
            )
        # a read-back order that never went in (e.g. submit_order refused): once, go back to it
        if not order and ud.pos.quote and not ud.transferred and not self._warned:
            self._warned = True
            raise ToolError(
                "The order was read back but never submitted. Fix what submit_order refused "
                "(ask the caller for anything missing) and submit it, unless they said they "
                "don't want it."
            )
        await end_after_reply(ctx)


class DeliveryUpdateAgent(Agent):
    """The outbound call: order #214 is out with the driver, and the Sprite ran out."""

    def __init__(self, pos: Pos) -> None:
        super().__init__(instructions=_outbound_instructions(pos))
        self._wrapped_at: float | None = None  # the read-back, or ending without the update

    async def on_enter(self) -> None:
        pos: Pos = self.session.userdata.pos
        # a fixed opener: who's calling and for whom, and nothing about the order until the
        # customer has said it's them
        self.session.say(
            f"Hi, this is {PERSONA} from {BUSINESS}, calling about a delivery. "
            f"Am I speaking with {first_name(pos.name or '')}?"
        )

    @function_tool
    async def swap_out_of_stock_item(self, ctx: Ctx, line_id: str, item_id: str) -> str:
        """Replace the out-of-stock item with another at the same price, once the customer
        picks it. Also when they change their mind: a different replacement, or the
        replacement after all once it was refunded.

        Args:
            line_id: The out-of-stock line, e.g. "L3".
            item_id: The replacement's menu id, e.g. "coke".
        """
        with _pos_rules(ctx) as pos:
            line = pos.swap_out_of_stock(line_id, item_id)
        return f"Swapped: {line.describe()}, same price. Total ${_total(pos)}.{_money(pos)}"

    @function_tool
    async def remove_out_of_stock_item(self, ctx: Ctx, line_id: str) -> str:
        """Take the out-of-stock item off the order and refund it, if the customer would
        rather not have a replacement, even after picking one earlier on this call.

        Args:
            line_id: The out-of-stock line, e.g. "L3".
        """
        with _pos_rules(ctx) as pos:
            line, refund = pos.remove_out_of_stock(line_id)
        return (
            f"Removed {line.describe()}. ${refund} goes back to their card (the item plus its "
            f"tax). New total ${_total(pos)}."
        )

    @function_tool
    async def add_to_delivery(self, ctx: Ctx, item_id: str, quantity: int = 1) -> str:
        """Add something the customer asks for to the order that's on its way, or change how
        many of it they want. Only items the driver carries can be added; the tool refuses
        anything else.

        Args:
            item_id: The menu id, e.g. "breadsticks".
            quantity: How many orders of it in all, not how many more; 0 takes it off.
        """
        with _pos_rules(ctx) as pos:
            line, charge = pos.add_extra(item_id, quantity)
        if not charge:
            return f"No change: it's already {line.describe()}. quantity is how many in all."
        if not quantity:
            return (
                f"Took the {line.item.name.lower()} off; they won't be charged the ${-charge}. "
                f"New total ${_total(pos)}."
            )
        more = f"${charge} more" if charge > 0 else f"${-charge} less"
        return (
            f"{line.describe()} on the order; the driver has them. {more} on the same card "
            f"(with tax). New total ${_total(pos)}."
        )

    @function_tool
    async def save_driver_note(self, ctx: Ctx, note: str) -> str:
        """Save drop-off details for the driver: a gate code, the building or unit, leave at
        the door, call on arrival. Pass the whole note each time; it replaces the last one.

        Args:
            note: The details, in a few words, e.g. "Gate code 4512, unit 3B, leave at door".
        """
        with _pos_rules(ctx) as pos:
            note = pos.set_driver_note(note)
        return f"Saved for {pos.delivery.driver if pos.delivery else 'the driver'}: {note}."

    @function_tool
    async def cancel_order(self, ctx: Ctx) -> str:
        """The customer wants to cancel the order. Call it before answering them, every time
        they ask."""
        with _pos_rules(ctx) as pos:
            pos.cancel()
        return "Cancelled."  # never reached: an order out for delivery can't be cancelled

    @function_tool
    async def note_for_manager(self, ctx: Ctx, summary: str) -> str:
        """Leave a note for the manager: a complaint, or a request for a call back. Only after
        the customer says they want it.

        Args:
            summary: What happened and what they want, in a sentence.
        """
        with _pos_rules(ctx) as pos:
            summary = pos.note_for_staff(summary)
        return f"Noted for the manager, who will call them back at {pos.phone}: {summary}."

    @function_tool
    async def final_read_back(self, ctx: Ctx) -> str:
        """Get the final changes and ETA to read back. Call it once nothing else is changing,
        before saying goodbye; call it again after any later change."""
        with _pos_rules(ctx) as pos:
            d = pos.read_back()
        changes = "; ".join(d.changes.values()) or "no changes"
        self._wrapped_at = time.time()
        return (
            f"Changes: {changes}. Order total ${_total(pos)}.{_money(pos)} Read back, in one or "
            "two sentences, each change, the total, and the refund or charge; then say "
            f"{d.driver} will be there in about {d.eta_minutes} minutes, and say goodbye. "
            "Then call end_call."
        )

    @function_tool
    async def end_without_update(self, ctx: Ctx, reason: EndedEarly) -> str:
        """End the call without going through the update, when whoever answered isn't the
        customer, can't talk now, or doesn't want to. The order stays as it is.

        Args:
            reason: wrong_person (a wrong number, or it isn't the customer), call_back_later
                (driving, busy, "call me later"), or declined (not interested).
        """
        with _pos_rules(ctx) as pos:
            d = pos.end_without_update(reason)
        self._wrapped_at = time.time()
        if reason == "wrong_person":
            return (
                "Recorded. Say a polite goodbye in a sentence (sorry for the mix-up, if it's a "
                "wrong number; thanks, if it's someone else who picked up), nothing more: don't "
                "mention the order, the address, the customer's name, or the driver, and don't "
                "ask who they are. Then call end_call."
            )
        return (
            f"Recorded; the order stays as it is, and {d.driver} is still on the way. Don't "
            "push about the order. Say goodbye in a sentence, and call end_call."
        )

    @function_tool
    async def end_call(self, ctx: Ctx) -> None:
        """Hang up, once you've read back the final changes and the ETA (or called
        end_without_update) and said goodbye."""
        ud = ctx.userdata
        await ctx.wait_for_playout()
        pos = ud.pos
        spoken = [
            m.created_at
            for m in ctx.session.history.messages()
            if m.role == "assistant" and m.text_content
        ]
        # the call isn't done until the customer has heard what changed and when it arrives,
        # unless it ended without the update
        if not pos.call_done or not any(t > (self._wrapped_at or 0) for t in spoken):
            raise ToolError(
                "Call final_read_back, read the changes and the ETA back to the customer, and "
                "say goodbye before ending the call. If it's the wrong person, or they can't or "
                "won't go through it now, call end_without_update instead."
            )
        d = pos.delivery
        await end_after_reply(
            ctx, ENDED_EARLY[d.ended_early] if d and d.ended_early else "Call complete"
        )


def _total(pos: Pos) -> str:
    return str(pos.quote.total) if pos.quote else "unknown"


def _money(pos: Pos) -> str:
    balance = pos.balance()
    if balance < 0:
        return f" ${-balance} is refunded to their card."
    if balance > 0:
        return f" ${balance} more goes on their card."
    return " Nothing more is charged or refunded."


def _voicemail() -> str:
    # who's calling, what about, and a number to call back: never the details
    number = spoken_number(RESTAURANT_NUMBER) if RESTAURANT_NUMBER else "the restaurant"
    return (
        f"Hi, this is {PERSONA} from {BUSINESS}, with a quick update on your delivery. Please "
        f"call us back at {number}. Thanks, goodbye."
    )


def _instructions(pos: Pos) -> str:
    menu = "\n".join(f"- {m.describe()}" for m in pos.menu)
    toppings = ", ".join(t.label for t in TOPPINGS)
    sauces = ", ".join(s.label for s in SAUCES)
    return f"""\
{VOICE_RULES}

You're {PERSONA}, answering the phone at {BUSINESS}, a neighborhood pizzeria in San
Francisco. You take pickup and delivery orders. Be warm and quick, like someone good at the
counter on a busy night.

# The menu (ids for the tools)
{menu}
Pizza toppings, $2 each: {toppings}.
Wing sauces: {sauces}. "Thai" means Thai chili peanut.
Only offer what's on this menu. Some items may be out today; the tools will tell you.
The ids (menu ids, and cart lines like L1) are for the tools only: never say them.
If they just ask what you have, give a quick idea in a sentence ("pizzas like pepperoni or
margherita, wings, salads, the usual sides") and ask what sounds good.
If they ask for the whole menu, read it all: call get_menu first, then go category by
category (pizzas, wings, sides, salads, drinks, desserts), naming every item that's available
today, in plain sentences with a short pause between categories. Don't stop partway to ask if
they want more, and don't skip items. Leave prices and sizes out unless they ask for them;
if they do, include them (pizzas come small, medium, or large). If they cut in to order or
ask about something, stop and help with that.
Say a price when they ask, and the total at the read-back.

# Taking the order
- Add each item with add_item as soon as the caller says it; don't wait for the whole order.
  Ask for anything required that's missing: pizza size, wing size and sauce, drink size.
- Confirm each item in a few words ("Large pepperoni, got it."), not the whole cart, and only
  after add_item has succeeded for it.
- People repeat themselves and talk in bursts. An item they name again that's already in
  the cart is the same item, restated: leave it alone. "Two large pepperonis" and later
  "yeah, two pepperonis" is still two, not four. A count goes up only when they say
  "another", "more", "add one", or a bigger total ("make it three"); then use update_item
  with the new total.
- Corrections: change exactly the line they mean with update_item or remove_item. Never
  remove and re-add, and never rebuild the cart.
- If an item is out, say so and offer one or two others from the same category.
- If they say something you can't match to the menu or a size, ask about just that part.
- Leaving something off a pizza ("no olives", "hold the onions"): pass it in `without`.
  The price doesn't change.
- Allergies: call items_with_allergen first. Name the items that contain it, per the menu.
  Never say anything is safe or free of an allergen; the kitchen is shared. For a severe
  allergy, offer to transfer them to a staff member (transfer_to_staff, once they say yes).
- Once, when the order sounds done and has no breadsticks, call offer_breadsticks and
  suggest them. That's the only upsell: don't push anything else.

# Finishing up
Don't ask for anything they've already told you (pickup or delivery, the address, the name,
the number), and when they give two things at once, take both.
1. If the order has no breadsticks and you haven't offered them, call offer_breadsticks
   and suggest them first. Then ask pickup or delivery. For delivery, get the full street address and call
   set_delivery; we deliver anywhere.
   Delivery is ${DELIVERY_FEE} and needs a ${DELIVERY_MINIMUM} subtotal.
2. Ask for a name, then (after they answer) a callback number, one at a time, and call
   set_contact with what you have.
3. Call price_order, then read back the whole order and the total in two or three
   sentences, and ask if that's right. Give the total with tax (and say it includes the
   delivery fee, for delivery); don't break out the subtotal and tax unless they ask.
4. If they change something during the read-back, make just that change, call price_order
   again, and say only what changed and the new total. Then ask if that's right.
5. When they say yes, call submit_order. Then, in your next words, tell them the order number
   and when it'll be ready, and ask if there's anything else. Only after that, say goodbye
   and call end_call; never in the same turn as submit_order."""


def _outbound_instructions(pos: Pos) -> str:
    d = pos.delivery
    out = pos.out_of_stock()
    assert d is not None and pos.order is not None and out
    lines = "; ".join(ln.describe() for ln in pos.lines)
    swaps = ", ".join(pos.same_price_swaps(out[0]))
    return f"""\
{VOICE_RULES}

You're {PERSONA} from {BUSINESS}, a pizzeria in San Francisco, making an outbound call to
{pos.name} about their delivery order. Be polite, clear, and brief.

# The order
Order #{pos.order.number}, paid by card: {lines}. Delivery to {pos.fulfillment.address}.
Total ${d.charged}. {d.driver}, the driver, is about {d.eta_minutes} minutes away.
The {out[0].item.name} ({out[0].id}) ran out after the order was placed.

# What to handle
You've already said who you are and asked for {first_name(pos.name or "")}. Once they say it's them,
tell them {d.driver} is about {d.eta_minutes} minutes away and that the {out[0].item.name} ran
out, and ask whether to replace it with a 2-liter Coke at the same price or remove it and refund
it.
If it isn't {pos.name} (a wrong number, or someone else who picked up), call
end_without_update with wrong_person and say a polite goodbye: say nothing more about the
order, the address, or who it's for.
If they're driving, busy, ask you to call later, or aren't interested, don't push: call
end_without_update (call_back_later or declined) and say goodbye.
Every change goes through its tool, in the same turn the customer asks for it. Never say a
change is made, or quote a total or refund, until its tool has returned.
1. The out-of-stock item. Replace it at the same price (swap_out_of_stock_item; same-price
   options: {swaps}), or remove it and refund it (remove_out_of_stock_item). Say the refund
   amount the tool returns. If they change their mind later, call the tool for the new
   choice.
2. Drop-off details. Ask if there's anything the driver should know: a gate code, the
   building or unit, leave at the door, call on arrival. Save what they say with
   save_driver_note. If there's nothing, don't call it; move on.
3. Adding an item: call add_to_delivery for whatever they ask for, and relay its answer.
   Breadsticks can be added (up to {MAX_EXTRAS} orders), since the driver carries them; say
   the added charge. To change how many, call it again with the new total (0 takes them
   off). Anything else is refused because the order is already on its way.
4. Cancelling: call cancel_order before you answer, every time they ask. It's too late once
   the driver is on the way: say so politely, and offer to leave a note for the manager to
   call them back (note_for_manager, only if they want it).
5. Everything else on the order is already with the driver and can't be changed.

# Wrapping up
When nothing else is changing, call final_read_back and read back what it returns: each
change, the order total, any refund or added charge, and the ETA. Then say goodbye and call
end_call. If anything changes after the read-back, call final_read_back again."""


server = AgentServer()


async def on_simulation_end(sim: SimulationContext) -> None:
    """Fails a run whose final POS state isn't what the scenario's `expect` says."""
    ud: Userdata = sim.job_context.primary_session.userdata
    if (expect := sim.userdata().get("expect")) and (problems := grade(ud.pos, expect)):
        sim.fail(reason="; ".join(problems))


@server.rtc_session(on_simulation_end=on_simulation_end)
async def entrypoint(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata or "{}")
    sim = ctx.simulation_context()
    if sim:
        # a simulated caller: the scenario's userdata says which direction
        meta = {**meta, **sim.userdata()}
    # console has no job metadata: CALL_DIRECTION=outbound picks the delivery update there
    outbound = (meta.get("direction") or os.getenv("CALL_DIRECTION")) == "outbound"
    ctx.log_context_fields = {
        "room": ctx.room.name,
        "direction": "outbound" if outbound else "inbound",
    }

    agent: Agent
    if outbound:
        # order #214 is already placed, paid for, and out with the driver
        pos = demo_pos("outbound", meta.get("name") or "")
        agent = DeliveryUpdateAgent(pos)
    else:
        # a restaurant doesn't know who's calling: Rosa asks for the name for the order
        pos = demo_pos()
        agent = RosaAgent(pos)

    tts = inference.TTS("cartesia/sonic-3.5", voice=VOICE, language="en")
    session = AgentSession[Userdata](
        userdata=Userdata(pos=pos),
        tts_text_transforms=["filter_markdown", "filter_emoji", pause_between_sentences],
        stt=inference.STT("deepgram/nova-3", language="en"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=tts,
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1"),
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": True},
        ),
    )

    await ctx.connect()
    if sim:
        await session.start(agent=agent, room=ctx.room, room_options=room_io.RoomOptions())
        return

    # the caller hung up, or the agent did: close the room, and the phone line with it
    session.on("close", lambda _: ctx.delete_room())
    dial = meta.get("dial")
    trunk = os.getenv("SIP_OUTBOUND_TRUNK_ID")
    if outbound and dial and trunk:
        await ring(ctx, session, agent, dial, trunk, tts)
        return
    if outbound and dial:
        logger.warning("SIP_OUTBOUND_TRUNK_ID isn't set; the person in the room plays the callee")
    # a phone call in, or someone in the room: on an outbound call, they play the customer
    caller = await ctx.wait_for_participant()
    await session.start(agent=agent, room=ctx.room, room_options=room_options(caller.identity))


async def ring(
    ctx: JobContext,
    session: AgentSession[Userdata],
    agent: Agent,
    number: str,
    trunk: str,
    tts: inference.TTS,
) -> None:
    """Rings a real phone; the agent joins only once a person has picked up."""
    # a listener of its own answers first, so nothing reaches the line until it's known who,
    # or what, picked up
    probe = AgentSession[None](tts=tts)
    await probe.start(agent=Agent(instructions=""), room=ctx.room, room_options=room_options(PHONE))
    async with AMD(probe, participant_identity=PHONE, ivr_detection=False) as detector:
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=trunk,
                    sip_call_to=number,
                    participant_identity=PHONE,
                    wait_until_answered=True,
                )
            )
        except api.ServerError as e:
            # busy, declined, no answer, or a trunk that can't place the call
            logger.warning("outbound call failed: %s", e)
            ctx.delete_room()
            return
        result = await detector.execute()
    logger.info("answered by %s", result.category, extra={"transcript": result.transcript})
    if result.category == "machine-vm":
        await probe.say(_voicemail(), allow_interruptions=False).wait_for_playout()
        await probe.aclose()
        await hang_up("Reached voicemail; left a message")
        return
    await probe.aclose()
    if result.category in ("machine-unavailable", "machine-ivr"):
        await hang_up("No one answered")
        return
    await session.start(agent=agent, room=ctx.room, room_options=room_options(PHONE))


if __name__ == "__main__":
    cli.run_app(server)
