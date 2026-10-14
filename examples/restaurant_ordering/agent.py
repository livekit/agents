"""Zappi's Pizzeria on the phone: Gabby takes pickup and delivery orders, and calls a customer
back when an item in their delivery runs out.

Calls are inbound by default. `{"direction": "outbound"}` in the job metadata (or
CALL_DIRECTION=outbound for `console`) places the delivery-update call instead: to a real phone
when the metadata also has `"dial"` and an outbound line (SIP_OUTBOUND_TRUNK_ID, or the
metadata's `trunk` or `from`), otherwise to whoever is in the room, who plays the customer.

The demo website watches calls through page.py; without it, the page hooks here do nothing.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal

from backend import (
    DELIVERY_FEE,
    ENDED_EARLY,
    MAX_EXTRAS,
    SAUCES,
    SMALL_ORDER_FEE,
    SMALL_ORDER_UNDER,
    TAX_RATE,
    TOPPINGS,
    EndedEarly,
    Line,
    Pos,
    PosError,
    demo_pos,
)
from call import (
    USER_AWAY_TIMEOUT,
    VOICE_RULES,
    end_after_reply,
    first_name,
    hang_up,
    hang_up_when_silent,
    room_options,
    spoken_number,
)
from dotenv import load_dotenv
from grade import grade
from page import Page, visitor_name
from scripted_callers import disconnect_scripted_callers

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

PERSONA = "Gabby"
BUSINESS = "Zappi's Pizzeria"
HOURS = "open every day, 11 a.m. to 10 p.m."
VOICE = "luna"  # Rime Coda "Luna"
# the number a customer calls back on, said on voicemail
RESTAURANT_NUMBER = os.getenv("RESTAURANT_NUMBER", "")
# the participant identity of a phone the agent rings
PHONE = "phone"


@dataclass
class Userdata:
    pos: Pos
    # the demo website's view of the call: inert unless a page is watching (page.py)
    page: Page = field(init=False)

    def __post_init__(self) -> None:
        self.page = Page(PERSONA, self.pos)


Ctx = RunContext[Userdata]


@contextlib.contextmanager
def _pos_rules(ctx: Ctx) -> Iterator[Pos]:
    """Run a POS call; a refusal shows on the page and goes back to the agent to relay."""
    try:
        yield ctx.userdata.pos
    except PosError as e:
        ctx.userdata.page.record(ctx.function_call.name, f"Refused: {str(e).split('. ')[0]}")
        raise ToolError(str(e)) from e


_CLAUSE = re.compile(r"[.,;!?]")
_TOKEN = re.compile(r"[a-z0-9]+")
# words that name some other kind of item, or a drink in general
_OTHER_KINDS = {"pizza", "pie", "wing", "salad", "side", "dessert"}
_DRINK_WORDS = {"soda", "drink", "pop"}


def _tokens(text: str) -> list[str]:
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in _TOKEN.findall(text.lower())]


def _drink_size_said(ctx: Ctx, pos: Pos, item_id: str, size: str | None) -> None:
    """Refuses a drink size the caller didn't give for a drink. Sodas and pizzas share
    small, medium, and large, so in "a large pepperoni and a Coke" the "large" is the
    pizza's, and the model tends to borrow it, or to pick a size, instead of asking."""
    item = pos.item(item_id)
    if item.category != "drinks" or not item.needs_size or not size:
        return
    chosen = item.size(size)
    sizes = {t for n in (chosen.id, chosen.label, *chosen.aliases) for t in _tokens(n)}
    sizes = {t for t in sizes if len(t) > 1}
    # any drink's name counts: in "make the large Coke a Sprite" the size is the Sprite's too
    drinks = [m for m in pos.menu if m.category == item.category]
    mine = {t for m in drinks for t in _tokens(m.name) if len(t) > 2} | _DRINK_WORDS
    others = (
        {t for m in pos.menu if m is not item for t in _tokens(m.name) if len(t) > 2} | _OTHER_KINDS
    ) - mine
    heard = [m.text_content or "" for m in ctx.session.history.messages() if m.role == "user"]
    for said in heard[-2:]:
        for clause in _CLAUSE.split(said):
            words = _tokens(clause)
            for i, w in enumerate(words):
                if w not in sizes:
                    continue
                # the item it goes with is the nearest one named in the same clause, the
                # one after it on a tie ("a large Coke"); none at all is an answer ("Small.")
                named = [(abs(j - i), j < i, x) for j, x in enumerate(words) if x in mine | others]
                if not named or min(named)[2] in mine:
                    return
    options = ", ".join(s.label for s in item.sizes)
    raise PosError(
        f"The caller hasn't said a size for the {item.name}: don't use another item's size or "
        f"pick one. Ask which, naming each one ({options})."
    )


def _line_note(pos: Pos, line: Line) -> str:
    # every allergy the caller mentioned stays in play for everything added after it
    if hits := sorted(line.allergens.intersection(pos.allergies)):
        return (
            f" Heads up: per the menu data this contains {' and '.join(hits)}, which the "
            "caller mentioned an allergy to. Tell them."
        )
    return ""


_PEPPERONIS = re.compile(r"\b(p)epperonis\b", re.IGNORECASE)


async def say_pepperoni_pizzas(text: AsyncIterable[str]) -> AsyncIterable[str]:
    """The voice mangles "pepperonis", so it says "pepperoni pizzas" instead."""
    buf = ""
    async for chunk in text:
        buf += chunk
        # hold back the last word, which may continue in the next chunk
        cut = max(buf.rfind(" "), buf.rfind("\n")) + 1
        if cut:
            yield _PEPPERONIS.sub(r"\1epperoni pizzas", buf[:cut])
            buf = buf[cut:]
    if buf:
        yield _PEPPERONIS.sub(r"\1epperoni pizzas", buf)


_MORE = re.compile(r"\b(more|another|add|extra|additional|back)\b")
# a yes to the agent's own "want one more?"
_YES = re.compile(r"^\W*(yes|yeah|yep|yup|sure|ok|okay|please|absolutely|definitely)\b")
_NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty"
).split()


def _asked_for_more(ctx: Ctx, total: int) -> bool:
    # whether the caller's last words ask for a bigger count ("two more", "another
    # one", "put the other one back"), state the new total ("make it three"), or say yes
    # to the agent offering one more
    said, offered = "", ""
    for m in reversed(ctx.session.history.messages()):
        if m.role == "user" and not said:
            said = (m.text_content or "").lower()
        elif m.role == "assistant" and said and m.text_content:
            offered = m.text_content.lower()
            break
    total_words = {str(total)} | ({_NUMBERS[total]} if total < len(_NUMBERS) else set())
    return (
        bool(_MORE.search(said))
        or any(re.search(rf"\b{w}\b", said) for w in total_words)
        or bool(_YES.match(said) and _MORE.search(offered))
    )


_NO = re.compile(r"^\W*(no|nope|nah|don'?t|do not|wait|hold on|not yet)\b")
# a read-back answered with a shrug isn't a yes: "sure, I guess", "whatever", "I don't know"
_UNSURE = re.compile(
    r"\b(whatever|i guess|i suppose|i don'?t know|i dunno|dunno|not sure|maybe|probably|"
    r"i think so|kind of|kinda)\b"
)
_AGREE = re.compile(
    r"\b(yes|yeah|yep|yup|yea|correct|right|sounds? good|looks? good|perfect|great|go ahead|"
    r"place it|do it|absolutely|definitely|exactly|all good|good|ok|okay|sure|confirm|"
    r"confirmed)\b"
)
_DISAGREE = re.compile(r"^\W*(no|nope|nah|wait|hold on)\b|\bnot (right|correct)\b|\bwrong\b")


def _read_back_answer(ctx: Ctx, since: float | None) -> str:
    """How the caller answered the read-back given at `since`: "yes", "unsure", "no", or
    "none" yet. The latest answer counts; words that are none of these (a phone number given
    after their yes) are skipped."""
    if since is None:
        return "none"
    heard = [
        (m.text_content or "").lower()
        for m in ctx.session.history.messages()
        if m.role == "user" and m.created_at > since
    ]
    for said in reversed(heard):
        if _DISAGREE.search(said):
            return "no"
        if _UNSURE.search(said):
            return "unsure"
        if _AGREE.search(said):
            return "yes"
    return "none"


def _answered_yes_since(ctx: Ctx, asked_at: float | None) -> bool:
    # whether the caller has answered since the cancellation was read back to them, and not
    # with a no: a cancel goes through only on their yes, never in the turn that asked
    if asked_at is None:
        return False
    heard = [
        m.text_content or ""
        for m in ctx.session.history.messages()
        if m.role == "user" and m.created_at > asked_at
    ]
    return bool(heard) and not _NO.match(heard[-1].lower())


def _cancel_read_back_at(ctx: Ctx) -> float | None:
    # when the agent asked "cancel the whole order?" in its own words, before calling
    # cancel_order: its last question before the caller's last words, if that was about
    # cancelling
    asked: float | None = None
    for m in ctx.session.history.messages():
        text = (m.text_content or "").lower()
        if m.role == "assistant" and text:
            asked = m.created_at if "cancel" in text and "?" in text else None
    return asked


def _cancel_needs_a_yes(ctx: Ctx, pos: Pos, confirmed: bool, asked_at: float | None) -> None:
    if confirmed and pos.cancel_asked and not _answered_yes_since(ctx, asked_at):
        raise PosError(
            "They haven't said yes to cancelling: wait for a clear yes to the read-back before "
            "calling cancel_order with confirmed. If they said no, carry on with the order."
        )


def _tip(amount: float | None, percent: float | None) -> dict[str, Decimal | None]:
    return {
        "amount": None if amount is None else Decimal(str(amount)),
        "percent": None if percent is None else Decimal(str(percent)),
    }


class GabbyAgent(Agent):
    def __init__(self, pos: Pos) -> None:
        super().__init__(instructions=_instructions(pos))
        self._submitted_at: float | None = None
        # when the current total was first read back, and for which cart version
        self._read_back_at: float | None = None
        self._read_back_version: int | None = None
        self._cancel_asked_at: float | None = None
        self._cancelled_at: float | None = None
        self._warned = False

    async def on_enter(self) -> None:
        self.session.userdata.page.push()
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
        ctx.userdata.page.record("get_menu", f"{len(pos.menu)} items")
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
        ctx.userdata.page.record("check_availability", f"{item.name}: available")
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
        ctx.userdata.page.record(
            "items_with_allergen", f"{tag}: {', '.join(items) or 'none listed'}"
        )
        found = (
            f"Per the menu data, these contain {tag}: {', '.join(items)}."
            if items
            else f"No item on the menu lists {tag}."
        )
        clear = [m.name for m in pos.menu if m.available and tag not in m.allergens]
        out = (
            f"{found} These don't list {tag}: {', '.join(clear) or 'nothing'}. But it's all made "
            f"in one shared kitchen, so you can't promise anything is free of {tag}. Never call "
            f'anything "safe", "safer", "safest", or "{tag}-free", in any form (not even "the '
            "safest thing is to skip it\": say \"I'd skip anything you're unsure of\"), and don't "
            'pick "the best choice" for them. Answer what they asked in a sentence or two; when a '
            'list is long, sum it up ("all the pizzas and the wings") instead of reading out '
            "every item."
        )
        if severe:
            out += (
                " It's a severe allergy: be serious and careful. You can't promise anything from "
                "a shared kitchen, so they should skip whatever they're unsure of. You handle it "
                "yourself: don't offer to pass them to anyone."
            )
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
        first_half: list[str] | None = None,
        second_half: list[str] | None = None,
    ) -> str:
        """Add a new item to the cart. Don't call it for an item that's already there: a
        caller naming it again is repeating themselves, not ordering more.

        Args:
            item_id: The menu id, e.g. "pepperoni".
            quantity: How many orders of it: "a dozen wings" is quantity 1, size dozen.
            size: Required for pizza (small, medium, large), wings (half dozen, dozen), and
                drinks (small, medium, large). Leave out for one-size items. Only a
                size the caller said for this item: never one carried over from another item
                in the same sentence ("a large pepperoni and a Coke": the Coke has no size
                yet), and never one you pick for them.
            toppings: Extra pizza toppings, e.g. ["mushrooms"]. Pizza only.
            sauce: Required for wings, e.g. "hot" or "Thai".
            without: What to leave off a pizza ("no olives"), e.g. ["olives"]. Same price.
            first_half: Half-and-half only ("half mushroom, half pepperoni"): the toppings on
                one half, e.g. ["mushrooms"]. One pizza, never two.
            second_half: The toppings on the other half, e.g. ["pepperoni"].
        """
        with _pos_rules(ctx) as pos:
            _drink_size_said(ctx, pos, item_id, size)
            line = pos.add(
                item_id, quantity, size, toppings, sauce, without, first_half, second_half
            )
        ctx.userdata.page.record("add_item", line.describe(), stage="order")
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
        first_half: list[str] | None = None,
        second_half: list[str] | None = None,
        item_id: str | None = None,
    ) -> str:
        """Change one line in the cart. Pass only what changed; the rest stays as it is.

        Args:
            line_id: The cart line, e.g. "L1".
            item_id: A different menu item for this line ("make the Coke a Sprite"), e.g.
                "sprite". It keeps the line's count, and its size if the new item comes in
                it: don't ask the size again.
            quantity: The new total for this line, not how many to add. "Two pepperonis"
                when the line already has two is a repeat: no change.
            size: The new size, only as the caller said it for this item: never one
                carried over from another item, and never one you pick for them.
            toppings: The full new list of extra toppings for this pizza.
            sauce: The new wing sauce.
            without: The full new list of what to leave off this pizza, e.g. ["olives"]; []
                puts everything back.
            first_half: The full new list of toppings on one half, for half-and-half.
            second_half: The full new list of toppings on the other half.
        """
        more = quantity is not None and _asked_for_more(ctx, quantity)
        with _pos_rules(ctx) as pos:
            current = pos.line(line_id)
            before = current.describe()
            # the size the line already has isn't a new one to check
            if not (size and current.size.matches(size)):
                _drink_size_said(ctx, pos, item_id or current.item.id, size)
            line = pos.update(
                line_id,
                quantity,
                size,
                toppings,
                sauce,
                without,
                more=more,
                first_half=first_half,
                second_half=second_half,
                item_id=item_id,
            )
        after = line.describe()
        if after == before:
            return f"No change: it's already {after}."
        was, now = before.split(": ", 1)[1], after.split(": ", 1)[1]
        ctx.userdata.page.record("update_item", f"{was} → {now}", stage="order")
        return f"Updated {after}.{_line_note(pos, line)} {pos.cart_summary()}"

    @function_tool
    async def remove_item(self, ctx: Ctx, line_id: str) -> str:
        """Take one line out of the cart.

        Args:
            line_id: The cart line, e.g. "L3".
        """
        with _pos_rules(ctx) as pos:
            line = pos.remove(line_id)
        ctx.userdata.page.record("remove_item", line.describe(), stage="order")
        return f"Removed {line.describe()}. {pos.cart_summary()}"

    @function_tool
    async def offer_breadsticks(self, ctx: Ctx) -> str:
        """Call this right before suggesting breadsticks, the only upsell, when the caller
        hasn't asked for them. It refuses if they were already offered or ordered, and then
        you mustn't suggest them. If they asked for breadsticks, call add_item instead."""
        heard = next(
            (
                m.text_content or ""
                for m in reversed(ctx.session.history.messages())
                if m.role == "user"
            ),
            "",
        )
        if re.search(r"\bbread ?sticks?\b", heard, re.IGNORECASE):
            # they asked for them: an order, not something to pitch back to them
            ctx.userdata.page.record("offer_breadsticks", "Refused: the caller asked for them")
            raise ToolError(
                "They just asked for breadsticks. Call add_item for them now, and don't ask "
                "whether to add them."
            )
        with _pos_rules(ctx) as pos:
            item = pos.offer_upsell()
        ctx.userdata.page.record("offer_breadsticks", "Upsell offered, once")
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
        ctx.userdata.page.record("set_pickup", "Pickup", stage="details")
        return "Pickup it is."

    @function_tool
    async def set_delivery(self, ctx: Ctx, address: str) -> str:
        """Set the order to delivery, to any full street address.

        Args:
            address: The street address as the caller said it, e.g. "300 De Haro St, 94107".
        """
        with _pos_rules(ctx) as pos:
            pos.set_delivery(address)
        ctx.userdata.page.record("set_delivery", address, stage="details")
        out = f"Delivery to {address}. The delivery fee is ${DELIVERY_FEE}."
        # the cart as it stands, so anything the caller asked for but isn't in it gets added
        return f"{out} {pos.cart_summary()}"

    @function_tool
    async def set_delivery_note(self, ctx: Ctx, note: str) -> str:
        """Save drop-off details for the driver on a delivery order: leave it at the door,
        don't ring the bell, a gate code, the unit. Pass the whole note each time; it replaces
        the last one.

        Args:
            note: The details, in a few words, e.g. "Leave at the door, don't ring the bell".
        """
        with _pos_rules(ctx) as pos:
            note = pos.set_delivery_note(note)
        ctx.userdata.page.record("set_delivery_note", note, stage="details")
        return f"Saved for the driver: {note}. It's read back with the order."

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
        contact = " · ".join(x for x in (pos.name, pos.phone) if x)
        ctx.userdata.page.record("set_contact", contact, stage="details")
        return f"Saved: name {pos.name or 'missing'}, phone {pos.phone or 'missing'}."

    # ---- read-back and submit ----

    @function_tool
    async def price_order(self, ctx: Ctx) -> str:
        """Work out the total for the cart as it stands, to read back to the caller. Call it
        again after any change: an order can only be submitted at a total that's current."""
        with _pos_rules(ctx) as pos:
            q = pos.price()
        if q.version != self._read_back_version:
            # a new total to read back: only an answer after this one places the order
            self._read_back_at, self._read_back_version = time.time(), q.version
        ctx.userdata.page.record("price_order", f"Total ${q.total}", stage="readback")
        out = f"{pos.cart_summary()} {q.describe()} (tax is {TAX_RATE:.0%} of the subtotal)."
        if note := pos.fulfillment.note:
            out += f" Note for the driver: {note}. Include it in the read-back."
        if q.small_order_fee:
            out += (
                f" Say the ${q.small_order_fee} small-order fee once in the read-back, since "
                f"it's under ${SMALL_ORDER_UNDER}; never suggest adding items to avoid it."
            )
        return f"{out} {pos.payment()}: only if they ask how to pay."

    @function_tool
    async def set_tip(
        self, ctx: Ctx, amount: float | None = None, percent: float | None = None
    ) -> str:
        """Add the tip the caller asks for, or change or remove it. Never suggest or ask for a
        tip. Pass one of amount or percent.

        Args:
            amount: The tip in dollars, e.g. 5 for "five dollars"; 0 takes it off.
            percent: A percent of the subtotal, e.g. 15 for "fifteen percent".
        """
        with _pos_rules(ctx) as pos:
            tip = pos.set_tip(**_tip(amount, percent))
        ctx.userdata.page.record("set_tip", f"Tip ${tip}", stage="readback")
        return (
            f"Tip is ${tip}, paid with the order in person. Call price_order, then read back the "
            "new total with the tip and ask if that's right."
        )

    @function_tool
    async def submit_order(self, ctx: Ctx) -> str:
        """Send the order to the kitchen. Only after you've read back the order and total
        from price_order and the caller clearly said it's right."""
        with _pos_rules(ctx) as pos:
            answer = _read_back_answer(ctx, self._read_back_at)
            if answer == "unsure":
                raise PosError(
                    "That wasn't a clear yes. Ask, in a few words: \"Just to be sure, should I "
                    'place the order?" Submit only on a clear yes.'
                )
            if answer == "no":
                raise PosError(
                    "They didn't say the read-back is right. Ask what to change, make the "
                    "change, and read back the new total."
                )
            if answer == "none" and pos.quote_current:
                raise PosError(
                    "They haven't said yes to the read-back yet. Read back the order and total "
                    "and ask if it's right; submit only after a clear yes."
                )
            order = pos.submit()
        detail = f"Order #{order.number}, {order.ready_in}"
        if order.note:
            detail += f" · driver note: {order.note}"
        ctx.userdata.page.record("submit_order", detail, stage="submitted")
        self._submitted_at = time.time()
        note = f" The driver has the note: {order.note}." if order.note else ""
        return (
            f"Order number {order.number} is in, {order.ready_in}.{note} Now tell the caller the "
            "order number and when it'll be ready, and ask if there's anything else. Don't "
            f"end the call yet. ({pos.payment()}.)"
        )

    @function_tool
    async def cancel_order(self, ctx: Ctx, confirmed: bool = False) -> str:
        """The caller wants to cancel their order. Call it first without confirmed: it tells
        you what to read back. Call it with confirmed=True only after they clearly say yes.

        Args:
            confirmed: True only once the caller said yes to the cancellation read-back.
        """
        with _pos_rules(ctx) as pos:
            if confirmed and not pos.cancel_asked and (at := _cancel_read_back_at(ctx)):
                # she already asked, before calling the tool: that was the read-back
                pos.cancel()
                self._cancel_asked_at = at
            _cancel_needs_a_yes(ctx, pos, confirmed, self._cancel_asked_at)
            result = pos.cancel(confirmed)
        if result == "not_placed":
            self._warned = True  # nothing to nag about at end_call: they don't want it
            ctx.userdata.page.record("cancel_order", "Nothing placed: no order")
            return (
                "Nothing has been placed, so there's nothing to cancel and nothing is due. Tell "
                "them no order went in, and ask if there's anything else; if not, say goodbye "
                "and call end_call."
            )
        assert pos.order is not None
        if result == "ask":
            self._cancel_asked_at = time.time()
            ctx.userdata.page.record("cancel_order", "Asked to confirm the cancellation")
            return (
                f'Read it back first: "Cancel the whole order, number {pos.order.number}? '
                'Nothing will be due." Then wait. Only on a clear yes, call cancel_order with '
                "confirmed=True; if not, carry on with the order as it is."
            )
        self._cancelled_at = time.time()
        ctx.userdata.page.record("cancel_order", f"Order #{pos.order.number} cancelled")
        return (
            f"Order {pos.order.number} is cancelled; nothing is due. Tell them, and ask if "
            "there's anything else."
        )

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
        if order and order.cancelled:
            # a cancelled order isn't done until the caller has heard that it is
            if not any(t > (self._cancelled_at or 0) for t in spoken):
                ud.page.record("end_call", "Refused: cancellation not told yet")
                raise ToolError(
                    f"Tell the caller order {order.number} is cancelled and nothing is due, "
                    "then say goodbye before ending the call."
                )
        # a submitted order isn't done until the caller has heard its number and pickup time
        elif order and not any(t > (self._submitted_at or 0) for t in spoken):
            ud.page.record("end_call", "Refused: order number not given yet")
            raise ToolError(
                f"Tell the caller their order number, {order.number}, and that it'll be "
                f"{order.ready_in}, then say goodbye before ending the call."
            )
        # a read-back order that never went in (e.g. submit_order refused): once, go back to it
        if not order and ud.pos.quote and not self._warned:
            self._warned = True
            ud.page.record("end_call", "Refused: order read back but not submitted")
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
        self._cancel_asked_at: float | None = None

    async def on_enter(self) -> None:
        ud: Userdata = self.session.userdata
        ud.page.push(stage="update")
        self.session.say(_opener(ud.pos))

    @function_tool
    async def swap_out_of_stock_item(self, ctx: Ctx, line_id: str, item_id: str) -> str:
        """Replace the out-of-stock item with another at the same price, once the customer
        picks it. Also when they change their mind: a different replacement, or the
        replacement after all once it was taken off.

        Args:
            line_id: The out-of-stock line, e.g. "L3".
            item_id: The replacement's menu id, e.g. "coke".
        """
        with _pos_rules(ctx) as pos:
            line = pos.swap_out_of_stock(line_id, item_id)
        ctx.userdata.page.record("swap_out_of_stock_item", line.describe(), stage="changes")
        return f"Swapped: {line.describe()}, same price.{_money(pos)}"

    @function_tool
    async def remove_out_of_stock_item(self, ctx: Ctx, line_id: str) -> str:
        """Take the out-of-stock item off the order, so it comes off what's due at the door,
        once the customer has said they'd rather not have a replacement (even after picking
        one earlier on this call). Never without their choice.

        Args:
            line_id: The out-of-stock line, e.g. "L3".
        """
        with _pos_rules(ctx) as pos:
            line, less = pos.remove_out_of_stock(line_id)
        detail = f"{line.describe()} · -${less} at the door"
        ctx.userdata.page.record("remove_out_of_stock_item", detail, stage="changes")
        return f"Removed {line.describe()} (the item plus its tax comes off).{_money(pos)}"

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
            line, diff = pos.add_extra(item_id, quantity)
        if not diff:
            return f"No change: it's already {line.describe()}. quantity is how many in all."
        sign = "+" if diff > 0 else "-"
        if not quantity:
            detail = f"Took off {line.item.name} · -${-diff} at the door"
            ctx.userdata.page.record("add_to_delivery", detail, stage="changes")
            return f"Took the {line.item.name.lower()} off.{_money(pos)}"
        detail = f"{line.describe()} · {sign}${abs(diff)} at the door"
        ctx.userdata.page.record("add_to_delivery", detail, stage="changes")
        change = f"${diff} more" if diff > 0 else f"${-diff} less"
        return (
            f"{line.describe()} on the order; the driver has them. That's {change} at the door "
            f"(with tax).{_money(pos)}"
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
        ctx.userdata.page.record("save_driver_note", note, stage="changes")
        return f"Saved for {pos.delivery.driver if pos.delivery else 'the driver'}: {note}."

    @function_tool
    async def set_tip(
        self, ctx: Ctx, amount: float | None = None, percent: float | None = None
    ) -> str:
        """Add the tip for the driver the customer asks for, or change or remove it; it's paid
        with the order at the door. Never suggest or ask for a tip. Pass one of amount or
        percent.

        Args:
            amount: The tip in dollars, e.g. 5 for "five dollars"; 0 takes it off.
            percent: A percent of the order's subtotal, e.g. 15 for "fifteen percent".
        """
        with _pos_rules(ctx) as pos:
            tip = pos.set_tip(**_tip(amount, percent))
        ctx.userdata.page.record("set_tip", f"Tip for the driver ${tip}", stage="changes")
        return f"Tip for the driver is ${tip}, paid at the door with the order.{_money(pos)}"

    @function_tool
    async def cancel_order(self, ctx: Ctx, confirmed: bool = False) -> str:
        """The customer wants to cancel the whole order. Call it first without confirmed: it
        tells you what to read back. Call it with confirmed=True only after they clearly say
        yes.

        Args:
            confirmed: True only once the customer said yes to the cancellation read-back.
        """
        with _pos_rules(ctx) as pos:
            if confirmed and not pos.cancel_asked and (at := _cancel_read_back_at(ctx)):
                # she already asked, before calling the tool: that was the read-back
                pos.cancel()
                self._cancel_asked_at = at
            _cancel_needs_a_yes(ctx, pos, confirmed, self._cancel_asked_at)
            result = pos.cancel(confirmed)
        d = pos.delivery
        assert d is not None and pos.order is not None
        if result == "ask":
            self._cancel_asked_at = time.time()
            ctx.userdata.page.record("cancel_order", "Asked to confirm the cancellation")
            return (
                f'Read it back first: "Cancel the whole order? {d.driver} will turn back, and '
                'nothing will be due." Then wait. Only on a clear yes, call cancel_order with '
                "confirmed=True; if not, carry on with the update."
            )
        ctx.userdata.page.record(
            "cancel_order", f"Order #{pos.order.number} cancelled", stage="changes"
        )
        return (
            f"Order {pos.order.number} is cancelled: {d.driver} is turning back, and nothing "
            "is due. Call final_read_back."
        )

    @function_tool
    async def final_read_back(self, ctx: Ctx) -> str:
        """Get the final changes and ETA to read back. Call it once nothing else is changing,
        before saying goodbye; call it again after any later change."""
        with _pos_rules(ctx) as pos:
            d = pos.read_back()
        changes = "; ".join(d.changes.values()) or "no changes"
        ctx.userdata.page.record("final_read_back", changes, stage="confirmed")
        self._wrapped_at = time.time()
        if pos.order and pos.order.cancelled:
            return (
                f"Order {pos.order.number} is cancelled; {d.driver} is turning back, and nothing "
                "is due. Say so in a sentence, sorry for the trouble, and say goodbye. Then call "
                "end_call."
            )
        return (
            f"Changes: {changes}.{_money(pos)} Read back, in one or two sentences, each change "
            "and what's due at the door (and how much more or less that is); then say "
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
        ctx.userdata.page.record("end_without_update", ENDED_EARLY[reason])
        self._wrapped_at = time.time()
        if reason == "wrong_person":
            return (
                "Recorded. Say a polite goodbye in a sentence (sorry for the mix-up, if it's a "
                "wrong number; thanks, if it's someone else who picked up), nothing more: don't "
                "mention the order, the address, the customer's name, or the driver, and don't "
                "ask who they are. Call end_call in that same turn, right after the goodbye."
            )
        return (
            f"Recorded; the order stays as it is, and {d.driver} is still on the way. Don't "
            "push about the order. Say goodbye in a sentence, and call end_call in that same "
            "turn, right after the goodbye."
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
        if pos.call_done and not any(t > (self._wrapped_at or 0) for t in spoken):
            ud.page.record("end_call", "Refused: no goodbye yet")
            raise ToolError(
                "Say your goodbye first, in a sentence, and call end_call again right after it, "
                "in the same turn."
            )
        if not pos.call_done:
            ud.page.record("end_call", "Refused: changes not read back yet")
            raise ToolError(
                "Call final_read_back, read the changes and the ETA back to the customer, and "
                "say goodbye before ending the call. If it's the wrong person, or they can't or "
                "won't go through it now, call end_without_update instead."
            )
        d = pos.delivery
        await end_after_reply(
            ctx, ENDED_EARLY[d.ended_early] if d and d.ended_early else "Call complete"
        )


def _money(pos: Pos) -> str:
    # what's due at the door now, against what was due when the order went out
    if pos.order and pos.order.cancelled:
        return " Nothing is due: the order is cancelled."
    balance, due = pos.balance(), pos.due_now()
    if balance < 0:
        return f" ${due} due at the door now, ${-balance} less than before; cash or card."
    if balance > 0:
        return f" ${due} due at the door now, ${balance} more than before; cash or card."
    return f" Still ${due} due at the door, cash or card."


def _opener(pos: Pos) -> str:
    # a fixed opener: who's calling and for whom, and nothing about the order until the
    # customer has said it's them. Without a name, it asks about the order instead: never a
    # made-up name
    if pos.name:
        return (
            f"Hi, this is {PERSONA} from {BUSINESS}, calling about a delivery. "
            f"Am I speaking with {first_name(pos.name)}?"
        )
    return (
        f"Hi, this is {PERSONA} from {BUSINESS}, calling about a delivery order. Did you place "
        "an order with us tonight?"
    )


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
Francisco. You take pickup and delivery orders. We're {HOURS} Anything else about the
restaurant that isn't in these instructions (the address, parking, other days or holidays,
how busy it is), you don't know: say so plainly, never make it up.
Sound warm, quick, and lightly upbeat, like a capable neighborhood pizzeria employee during a
busy shift: welcoming and helpful, never bubbly or salesy (about a 6 out of 10 for
enthusiasm). For example: "Absolutely, I can help with that. What size would you like?" Drop
the upbeat tone and be calm, plain, and careful whenever an allergy comes up, or a
complaint, or a cancellation.

# The menu (ids for the tools)
{menu}
Pizza toppings, $2 each ($1 on half a pizza): {toppings}.
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
  Ask for anything required that's missing for what they ordered: pizza size, wing size and
  sauce, the size of a drink they asked for, naming the choices ("Small, medium, or large?").
  Never ask about or suggest anything they haven't mentioned (a drink, a side, an extra):
  the breadsticks offer is the only suggestion.
- Half-and-half ("half mushroom, half pepperoni"): one pizza, a cheese pizza unless they
  name another, with each half's toppings in first_half and second_half. Don't make two
  pizzas or put both on the whole pizza. Two different specialty pizzas on one pie isn't
  something we do: offer half-and-half toppings instead.
  Pass only a size the caller said for that item. Never carry one over from another item in
  the same sentence ("a large pepperoni and a Coke": the Coke has no size yet, so ask), and
  never pick one for them.
- Say "pepperoni pizzas", never "pepperonis".
- Confirm each item in a few words ("Large pepperoni, got it."), not the whole cart, and only
  after add_item has succeeded for it.
- People repeat themselves and talk in bursts. An item they name again that's already in
  the cart is the same item, restated: leave it alone. "Two large pepperonis" and later
  "yeah, two pepperonis" is still two, not four. A count goes up only when they say
  "another", "more", "add one", "put the other one back", a bigger total ("make it
  three"), or yes when you offered one more; then use update_item with the new total.
- Corrections: change exactly the line they mean with update_item or remove_item. Never
  remove and re-add, and never rebuild the cart. To change what an item is ("make the Coke
  a Sprite"), pass the new item_id to update_item: it keeps the count and the size, so don't
  ask the size again.
- If an item is out, say so and offer one or two others from the same category.
- If they say something you can't match to the menu or a size, ask about just that part.
- Leaving something off a pizza ("no olives", "hold the onions"): pass it in `without`.
  The price doesn't change.
- Allergies: call items_with_allergen first. Name the items that contain it, per the menu.
  Never use the words "safe", "safer", or "safest" in any form, even when they do ("the
  safest thing is to skip..."): say "I'd skip anything you're unsure of" instead. Never call
  anything free of an allergen or "the best choice" for an allergy; the kitchen is shared. A
  severe allergy gets the same answer, said seriously.
- Something we don't do (cutting it in squares, extra crispy, a crust or item
  that isn't on the menu): say it the way someone at the counter would, briefly and kindly
  ("Sorry, we don't do that, but I can..."), with the closest thing we do have, if there is
  one. Never mention "the system", "options", or what you can or can't see.
- Discount codes and coupons: we don't have any. Say so in a sentence ("Sorry, we don't
  have any discount codes right now.") and carry on.
- You handle the whole call yourself: never offer to pass them, transfer them, or leave a
  note for a staff member or a manager, and never promise a call back. If they ask for a
  person, say you're the one taking orders and can help with it.
- Never say you've noted, saved, passed on, or done something unless a tool did it, and
  never say you'll check or look into something you have no tool for. If there's no tool
  for what they ask, say plainly that you can't do that.
- Payment: they pay in person, cash or card: when they pick it up, or to the driver at the
  door. Answer "can I pay cash / by card / by text / online?" from that alone: you can't
  take payment over the phone or send a payment link. Never ask for card details; if they
  start to read out a card number, stop them politely ("No need, you'll pay in person when
  you get it.").
- Tips: never suggest or ask for one. If they want to add a tip (for the driver, or the
  kitchen), call set_tip with the dollars or percent they say; it's paid with the order. A
  tip change after the read-back is a change like any other: price and read back again.
- Drop-off details for a delivery ("leave it at the door", "don't ring the bell", a gate
  code, the unit): save them with set_delivery_note. Pickup has no driver to pass a note to.
- Cancelling: before the order is submitted, "never mind" or "cancel it" just means nothing
  goes in: call cancel_order, which tells you so. After submit_order, they can cancel the
  order they placed on this call: call cancel_order, read back what it tells you ("Cancel
  the whole order? Nothing will be due."), and only on a clear yes call it again with
  confirmed=True.
- A complaint, or "I want something for my trouble": apologize sincerely, and offer only
  what you can actually do: change or swap an item, take one off, or cancel. Whether to
  tip is up to them. Promise nothing else: no discounts, credits, free items, refunds, or
  call backs.
- Once, when the order sounds done and has no breadsticks, call offer_breadsticks and
  suggest them. That's the only upsell: don't push anything else. If they ask for
  breadsticks themselves, that's an order: add_item, no offer.
- Whatever they ask for, add it with add_item right away. Never ask "Should I add that?"
  about something they just ordered; ask only for what's missing (a size, a sauce).

# Finishing up
Don't ask for anything they've already told you (pickup or delivery, the address, the name,
the number), and when they give two things at once, take both.
1. If the order has no breadsticks and you haven't offered them, call offer_breadsticks
   and suggest them first. Then ask pickup or delivery. For delivery, get the full street
   address and call set_delivery; we deliver anywhere, for ${DELIVERY_FEE}.
   An order under ${SMALL_ORDER_UNDER} (pickup or delivery) has a ${SMALL_ORDER_FEE}
   small-order fee. Mention it once in the read-back; never push them to add items to avoid
   it.
2. Ask for a name, then (after they answer) a callback number, one at a time, and call
   set_contact with what you have.
3. Call price_order, then read back the whole order and the total in two or three
   sentences, and ask if that's right. Give the total with tax, and say what it includes: the
   delivery fee, for delivery; the small-order fee, if there is one ("plus a
   ${SMALL_ORDER_FEE} small-order fee, since it's under ${SMALL_ORDER_UNDER:.0f}"); the tip,
   if they added one. Read back the note for the driver too, if there is one. Don't break
   out the subtotal and tax unless they ask.
4. If they change something during the read-back, make just that change, call price_order
   again, and say only what changed and the new total. Then ask if that's right.
5. Only a clear yes ("yes", "that's right", "sounds good") places the order. "Sure, I
   guess", "whatever", or "I don't know" isn't one: ask "Just to be sure, should I place
   it?" When they clearly say yes, call submit_order. Then, in your next words, tell them
   the order number and when it'll be ready, and ask if there's anything else. Only after
   that, say goodbye and call end_call; never in the same turn as submit_order."""


def _outbound_instructions(pos: Pos) -> str:
    d = pos.delivery
    out = pos.out_of_stock()
    assert d is not None and pos.order is not None and out
    lines = "; ".join(ln.describe() for ln in pos.lines)
    swaps = ", ".join(pos.same_price_swaps(out[0]))
    if pos.name:
        to = pos.name
        reached = f"asked for {first_name(pos.name)}. Once they say it's them,"
        not_them = f"it isn't {pos.name}"
    else:
        # no name to go on: never one made up
        to = "the customer who placed it"
        reached = """asked whether they placed an order with us
tonight. You don't have their name: never say or guess one, and don't ask for it; if they
give it, use it. Once they say yes, it was their order,"""
        not_them = "they didn't place an order"
    return f"""\
{VOICE_RULES}

You're {PERSONA} from {BUSINESS}, a pizzeria in San Francisco, making an outbound call to
{to} about their delivery order. Be polite, clear, and brief, warm but not bubbly.
Drop the upbeat tone and be calm and plain about anything to do with money (what's due at
the door, more or less than before), a complaint, a cancellation, or an allergy.

# The order
Order #{pos.order.number}: {lines}. Delivery to {pos.fulfillment.address}.
{BUSINESS} is {HOURS}
${d.due} is due at the door: they pay {d.driver}, the driver, cash or card, when it arrives.
{d.driver} is about {d.eta_minutes} minutes away.
The {out[0].item.name} ({out[0].id}) ran out after the order was placed.

# What to handle
You've already said who you are and {reached}
tell them {d.driver} is about {d.eta_minutes} minutes away and that the {out[0].item.name} ran
out, and ask whether to replace it with a large Coke at the same price
or take it off the order (it comes off what they pay at the door).
Once they've said it's them, it's their order: when they ask, read them what's on it, the
address, and what's due, plainly.
If {not_them} (a wrong number, or someone else who picked up), call
end_without_update with wrong_person and say a polite goodbye: say nothing more about the
order, the address, or who it's for.
If they're driving, busy, ask you to call later, or aren't interested, don't push: call
end_without_update (call_back_later or declined), then say goodbye and call end_call in that
same turn.
Every change goes through its tool, in the same turn the customer asks for it. Never say a
change is made, or quote an amount, until its tool has returned. Swap the item, take it
off, or cancel only when the customer has clearly chosen that: never pick for them, and if
it's unclear, ask. Never say you've noted, saved, passed on, updated, or done something
unless a tool did it, and never say you'll check or look into something you have no tool
for; if there's no tool for it, say plainly that you can't do that. Anything about the
restaurant that isn't in these instructions, you don't know: say so, never make it up.
1. The out-of-stock item. Replace it at the same price (swap_out_of_stock_item; same-price
   options: {swaps}), or take it off the order (remove_out_of_stock_item): say how much less
   is due at the door, as the tool gives it. If they change their mind later, call the tool
   for the new choice.
2. Drop-off details. Ask if there's anything the driver should know: a gate code, the
   building or unit, leave at the door, call on arrival. Save what they say with
   save_driver_note. If there's nothing, don't call it; move on. The address can't change:
   {d.driver} takes it to the address on the order. If they need it somewhere else, they
   can cancel.
3. Adding an item: call add_to_delivery for whatever they ask for, and relay its answer.
   Breadsticks can be added (up to {MAX_EXTRAS} orders), since the driver carries them; say
   how much more that is at the door. To change how many, call it again with the new total
   (0 takes them off). Anything else is refused because the order is already on its way.
4. A tip for the driver: never suggest or ask for one. If they want to add one, call
   set_tip with the dollars or percent they say; it's added to what's due at the door.
5. Cancelling: they can cancel the whole order, even though it's on its way: {d.driver}
   turns back and nothing is due. Call cancel_order, read back what it tells you ("Cancel the
   whole order? {d.driver} will turn back, and nothing will be due."), and only on a clear
   yes call it again with confirmed=True.
6. Everything else on the order is already with the driver and can't be changed.
Payment is only in person, at the door, cash or card: answer "can I pay cash / by card / by
text?" from that alone. Never ask for card details; if they start to read out a card number,
stop them politely ("No need, you'll pay {d.driver} at the door."). Nothing is charged or
refunded over the phone: a change just makes what's due at the door more or less.
A complaint, or "I want something for my trouble": apologize sincerely, and offer only what
you can actually do: the swap, taking the out-of-stock item off, or cancelling. Whether to
tip is up to them. Promise nothing else: no discounts, credits, free items, refunds, or call
backs. You handle the whole call yourself: never offer to pass them, transfer them, or leave
a note for a manager or anyone else.

# Wrapping up
When nothing else is changing, call final_read_back and read back what it returns: each
change, what's due at the door, and the ETA (or, if it's cancelled, that nothing is due).
Then say goodbye and call end_call. If anything changes after the read-back, call
final_read_back again."""


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

    await ctx.connect()
    dial = None if sim else meta.get("dial")
    trunk, from_number = _outbound_line(meta)
    ring_phone = bool(outbound and dial and (trunk or from_number))
    if outbound and dial and not ring_phone:
        logger.warning(
            "no outbound line (SIP_OUTBOUND_TRUNK_ID); the person in the room plays the callee"
        )
    caller = None
    if not sim and not ring_phone:
        # a phone call in, or someone in the room: on an outbound call, they play the customer
        caller = await ctx.wait_for_participant()

    agent: Agent
    if outbound:
        # order #214 is already placed, paid for, and out with the driver, under the name the
        # website's visitor gave (or the metadata's `name`); with neither, it's under no name,
        # and Gabby asks about the order instead of for someone
        pos = demo_pos("outbound", visitor_name(caller) or meta.get("name") or "")
        agent = DeliveryUpdateAgent(pos)
    else:
        # a restaurant doesn't know who's calling: Gabby asks for the name for the order
        pos = demo_pos()
        agent = GabbyAgent(pos)

    tts = inference.TTS("rime/coda", voice=VOICE, language="en")
    session = AgentSession[Userdata](
        userdata=Userdata(pos=pos),
        tts_text_transforms=[
            "filter_markdown",
            "filter_emoji",
            say_pepperoni_pizzas,
        ],
        stt=inference.STT("deepgram/nova-3", language="en"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=tts,
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(version="v1"),
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": True},
        ),
        # neither side speaking this long makes the caller "away": see hang_up_when_silent
        user_away_timeout=USER_AWAY_TIMEOUT,
    )

    if sim:
        await session.start(agent=agent, room=ctx.room, room_options=room_io.RoomOptions())
        return

    # the caller hung up, or the agent did: close the room, and the phone line with it
    session.on("close", lambda _: ctx.delete_room())
    # a caller who's gone silent gets one "are you still there?", then a goodbye
    hang_up_when_silent(session, session.userdata.page)
    # a bot farm runs known question banks at public agents: hang up on their opening lines
    disconnect_scripted_callers(session, session.userdata.page)
    # the demo website, if it's watching: the page that asked to ring a phone, the browser
    # caller, or viewers of a call to the restaurant's number
    session.userdata.page.watch(ctx.room, session, page=meta.get("page"), caller=caller)
    if dial and ring_phone:
        await ring(ctx, session, agent, dial, trunk, from_number, tts)
        return
    assert caller is not None
    await session.start(agent=agent, room=ctx.room, room_options=room_options(caller.identity))


def _outbound_line(meta: dict[str, str]) -> tuple[str, str]:
    """The line a call to a real phone goes out on, as (trunk, from): the metadata's (a SIP
    trunk, or a number of the project's own to dial from), else SIP_OUTBOUND_TRUNK_ID."""
    if meta.get("trunk") or meta.get("from"):
        return meta.get("trunk") or "", meta.get("from") or ""
    return os.getenv("SIP_OUTBOUND_TRUNK_ID", ""), ""


async def ring(
    ctx: JobContext,
    session: AgentSession[Userdata],
    agent: Agent,
    number: str,
    trunk: str,
    from_number: str,
    tts: inference.TTS,
) -> None:
    """Rings a real phone; the agent joins only once a person has picked up."""
    page = session.userdata.page
    # a listener of its own answers first, so nothing reaches the line until it's known who,
    # or what, picked up
    probe = AgentSession[None](tts=tts)
    await probe.start(agent=Agent(instructions=""), room=ctx.room, room_options=room_options(PHONE))
    async with AMD(probe, participant_identity=PHONE, ivr_detection=False) as detector:
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    # a carrier's trunk reaches any country; without one, LiveKit dials out
                    # from one of the project's own numbers, US only
                    sip_trunk_id=trunk,
                    sip_number=from_number,
                    sip_call_to=number,
                    participant_identity=PHONE,
                    wait_until_answered=True,
                )
            )
        except api.TwirpError as e:
            # busy, declined, no answer, or a line that can't place the call
            logger.warning("outbound call failed: %s", e)
            await page.dial_failed(e, number)
            ctx.delete_room()
            return
        result = await detector.execute()
    logger.info("answered by %s", result.category, extra={"transcript": result.transcript})
    if result.category == "machine-vm":
        await probe.say(_voicemail(), allow_interruptions=False).wait_for_playout()
        await probe.aclose()
        await hang_up("Reached voicemail; left a message", page)
        return
    await probe.aclose()
    if result.category in ("machine-unavailable", "machine-ivr"):
        await hang_up("No one answered", page)
        return
    await session.start(agent=agent, room=ctx.room, room_options=room_options(PHONE))


if __name__ == "__main__":
    cli.run_app(server)
