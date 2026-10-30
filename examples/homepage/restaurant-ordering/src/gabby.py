"""Gabby at Zappi's Pizzeria: taking orders on the phone, and calling a customer whose delivery is
about to go out when something in it ran out.

Tools return facts and the next step; how to say things lives in the instructions, once. Placing,
cancelling, or dropping an order is read back first and done only after the caller has heard the
read-back in full and answered it (conversation.py)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.llm import ToolError
from pydantic import BaseModel, Field

from conversation import answered, caller_spoke_last, heard
from pizzeria import (
    CRUST,
    DRINK_SIZES,
    HOURS,
    MENU,
    NEIGHBORHOOD,
    PIZZA_SIZES,
    RESTAURANT,
    SAUCES,
    SOLD_OUT_TODAY,
    TOPPINGS,
    Allergen,
    EndedEarly,
    Ingredient,
    ItemId,
    Pizzeria,
    PizzeriaError,
    SauceId,
    Size,
    ToppingId,
    compared,
    money,
    size_words,
)

PERSONA = "Gabby"
GREETING = f"Thanks for calling {RESTAURANT}, this is {PERSONA}. What can I get started for you?"

VOICE_RULES = """\
You're on a phone call, so talk like a person, not a document:
- One or two short sentences at a time, and one question at a time. No lists, markdown, or symbols.
- Say prices and times the way people say them ("nineteen ninety-nine", "about twenty minutes"). Never read out ids.
- Only state a price, total, or order number that a tool gave you.
- Never say you've done something, or that you're about to, without calling its tool in that same reply.
- When a tool refuses, fix what it says with your tools. Tell the caller only what changes for them.
- You handle everything yourself. There's nobody to transfer to, and no callbacks to promise.
- They pay in person, cash or card, never over the phone: never take card details. There are no discount codes.
- A tip only if they ask to add one: set_tip.
- When they're done, say a short goodbye and call end_call in that same reply."""


class OrderItem(BaseModel):
    item_id: ItemId = Field(description="The menu id.")
    size: Size | None = Field(
        None,
        description="Required for pizzas, wings, and drinks that come in sizes, and only a "
        "size the caller said for this item.",
    )
    quantity: int = Field(1, description='How many: "a dozen wings" is 1, size dozen.')
    toppings: list[ToppingId] = Field(
        [], description="Extra toppings on a whole pizza."
    )
    sauce: SauceId | None = Field(
        None,
        description="Wings only, and only a sauce the caller chose from ours. If they haven't "
        "chosen, or asked for one we don't have, leave it out: the wings still go on, and "
        "you ask.",
    )
    without: list[Ingredient] = Field(
        [], description='What to leave off a pizza ("no olives"), from its ingredients.'
    )
    first_half: list[ToppingId] = Field(
        [], description="Half-and-half only: the toppings on one half."
    )
    second_half: list[ToppingId] = Field(
        [], description="Half-and-half only: the toppings on the other half."
    )


@dataclass(frozen=True)
class Review:
    """Something read back to the caller before it's done: placing or cancelling the order."""

    action: str
    revision: int
    token: str


@contextmanager
def _pos_rules() -> Iterator[None]:
    try:
        yield
    except PizzeriaError as e:
        raise ToolError(str(e)) from e


def _facts(**fields: Any) -> str:
    return json.dumps(fields)


class Gabby(Agent):
    """What both of Gabby's calls share: the order, read-backs, a tip, cancelling, and hanging
    up."""

    def __init__(self, pos: Pizzeria, *, instructions: str) -> None:
        self._pos = pos
        self._review: Review | None = None
        # what the caller must hear before the call can end, and the call that said it
        self._must_hear: dict[str, str] = {}
        super().__init__(instructions=instructions)

    def _answered_review(self, ctx: RunContext, action: str) -> bool:
        r = self._review
        return (
            r is not None
            and (r.action, r.revision) == (action, self._pos.revision)
            and answered(ctx.session.history.items, r.token)
        )

    def _read_back(self, ctx: RunContext, action: str) -> None:
        self._review = Review(action, self._pos.revision, ctx.function_call.call_id)

    def _read_back_status(self) -> dict[str, str]:
        """After a change to something already read back: get the new read-back first."""
        r = self._review
        # a change after reading back what dropping the order takes off means they kept it
        if r is None or r.action == "drop" or r.revision == self._pos.revision:
            return {}
        tool = {"place": "place_order", "cancel": "cancel_order"}[r.action]
        return {
            "readBack": f"Out of date: this changed what you read back. Call {tool} for the "
            "new read-back and read it to them before they agree."
        }

    def _must_tell(self, ctx: RunContext, what: str) -> None:
        # the latest outcome is the one they need: a cancellation supersedes the order number
        self._must_hear = {what: ctx.function_call.call_id}

    @function_tool
    async def set_tip(
        self,
        ctx: RunContext,
        dollars: float | None = None,
        percent: float | None = None,
    ) -> str:
        """Add the tip the caller asks for, or change it; 0 takes it off. Only when they ask.
        Pass one of the two.

        Args:
            dollars: The tip in dollars, e.g. 5.
            percent: A percent of the food, e.g. 15.
        """
        with _pos_rules():
            tip = await self._pos.set_tip(dollars, percent)
        return _facts(
            **{**self._money_facts(), "tip": money(tip)}, **self._read_back_status()
        )

    @function_tool
    async def cancel_order(self, ctx: RunContext) -> str:
        """Cancel the whole order. Call it the moment they ask, before you ask them anything:
        the first call returns what to confirm with them. Call it again once they agree."""
        if self._pos.order is None:
            return await self._drop_unplaced(ctx)
        if self._answered_review(ctx, "cancel"):
            with _pos_rules():
                order = await self._pos.cancel()
            self._must_tell(ctx, "that the order is cancelled and nothing is due")
            return _facts(
                cancelled=f"order {order.number}",
                due=money(self._pos.due_now()),
                next="Tell them it's cancelled and nothing is due, and ask if there's anything "
                "else.",
            )
        with _pos_rules():
            read_back = self._pos.cancel_read_back()
        self._read_back(ctx, "cancel")
        return _facts(
            cancelled=False,
            readBack=read_back,
            next="It isn't cancelled yet. Ask them now, using this, whether to cancel the "
            "whole order, and wait for their answer. Call cancel_order again only once they "
            "agree.",
        )

    async def _drop_unplaced(self, ctx: RunContext) -> str:
        """Cancelling before anything's placed: there's nothing to cancel, but what they've
        ordered so far comes off only once they've agreed to drop all of it."""
        if not self._pos.lines:
            self._must_tell(ctx, "that nothing was placed, so nothing is due")
            return _facts(
                cancelled=False,
                placed=False,
                next="Nothing was placed, so there's nothing to cancel and nothing is due. "
                "Tell them.",
            )
        if self._answered_review(ctx, "drop"):
            with _pos_rules():
                await self._pos.drop()
            self._must_tell(ctx, "that the order is dropped and nothing is due")
            return _facts(
                dropped=True,
                due=money(self._pos.due_now()),
                next="Tell them it's dropped and nothing is due. If they've already said "
                "goodbye, say goodbye and end_call in this same reply; otherwise ask if "
                "there's anything else.",
            )
        with _pos_rules():
            read_back = self._pos.drop_read_back()
        self._read_back(ctx, "drop")
        return _facts(
            dropped=False,
            readBack=read_back,
            next="Nothing's placed yet, and nothing comes off until they agree. If they might "
            "mean one item, or something about one item, ask which. Otherwise ask them now, "
            "using this, whether to drop the whole order, and wait for their answer. Call "
            "cancel_order again only once they agree to drop all of it.",
        )

    @function_tool
    async def end_call(self, ctx: RunContext) -> str | None:
        """Hang up. Only once the caller has nothing else, and after a short goodbye in this
        same reply."""
        # what this same reply says lands in the history once it has played
        await ctx.wait_for_playout()
        items = ctx.session.history.items
        if unheard := [
            w for w, token in self._must_hear.items() if not heard(items, token)
        ]:
            self._must_hear[unheard[0]] = ctx.function_call.call_id
            raise ToolError(f"Don't hang up yet: first tell them {unheard[0]}.")
        if missing := self._unfinished():
            raise ToolError(f"Don't hang up yet: {missing}")
        # the call ends once this reply has played, a goodbye included
        ctx.speech_handle.add_done_callback(lambda _: ctx.session.shutdown())
        if caller_spoke_last(items):
            return "Say a short goodbye."
        return None

    def _unfinished(self) -> str | None:
        return None

    def _money_facts(self) -> dict[str, Any]:
        quote = self._pos.quote()
        return quote.facts() if quote else {}


class TakeOrder(Gabby):
    """Gabby answering the phone: takes the order, how it gets to them, and who it's for, reads
    it back, and sends it to the kitchen."""

    def __init__(self, pos: Pizzeria) -> None:
        super().__init__(pos, instructions=_order_instructions())

    async def on_enter(self) -> None:
        self.session.say(GREETING)

    @function_tool
    async def add_items(self, ctx: RunContext, items: list[OrderItem]) -> str:
        """Put what the caller orders on the order as soon as they say it: everything they just
        named, in one call. Anything that can't go on yet comes back with what to ask.

        Args:
            items: Each item they named, as they asked for it.
        """
        with _pos_rules():
            added, problems = await self._pos.add_items(
                [item.model_dump() for item in items]
            )
        noted = sorted(self._pos.allergies)
        return _facts(
            added=[ln.facts() for ln in added],
            notAdded=problems,
            containsAllergyTheyMentioned=[
                {"item": ln.label, "contains": sorted(ln.allergens & set(noted))}
                for ln in added
                if ln.allergens & set(noted)
            ],
            **self._pos.cart_facts(),
            **self._read_back_status(),
        )

    @function_tool
    async def update_item(
        self,
        ctx: RunContext,
        line_id: str,
        item_id: ItemId | None = None,
        size: Size | None = None,
        quantity: int | None = None,
        toppings: list[ToppingId] | None = None,
        sauce: SauceId | None = None,
        without: list[Ingredient] | None = None,
        first_half: list[ToppingId] | None = None,
        second_half: list[ToppingId] | None = None,
    ) -> str:
        """Change one line on the order. Pass only what changes; everything else stays as it
        is.

        Args:
            line_id: The line, e.g. "L1".
            item_id: A different menu item for this line ("make the Coke a Sprite").
            size: The new size.
            quantity: The new total for this line, not how many more.
            toppings: The full new list of extra toppings; [] takes them all off.
            sauce: The new wing sauce.
            without: The full new list of what to leave off; [] puts everything back.
            first_half: The full new list for one half of a half-and-half.
            second_half: The full new list for the other half.
        """
        with _pos_rules():
            line, changed = await self._pos.update_item(
                line_id,
                item_id=item_id,
                size=size,
                quantity=quantity,
                toppings=toppings,
                sauce=sauce,
                without=without,
                first_half=first_half,
                second_half=second_half,
            )
        return _facts(
            **(
                {"updated": line.facts()}
                if changed
                else {"nothingChanged": f"{line.id} is already {line.describe()}."}
            ),
            **self._pos.cart_facts(),
            **self._read_back_status(),
        )

    @function_tool
    async def remove_item(self, ctx: RunContext, line_id: str) -> str:
        """Take one line off the order.

        Args:
            line_id: The line, e.g. "L3".
        """
        with _pos_rules():
            line = await self._pos.remove_item(line_id)
        return _facts(
            removed=line.describe(),
            **self._pos.cart_facts(),
            **self._read_back_status(),
        )

    @function_tool
    async def check_allergen(
        self, ctx: RunContext, allergen: Allergen, severe: bool = False
    ) -> str:
        """Look up which menu items contain an allergen, before answering any allergy
        question. One allergen per call. For one the menu doesn't track, say so.

        Args:
            allergen: The allergen. Lactose intolerance is dairy, celiac is gluten.
            severe: Only if they said it's severe.
        """
        with _pos_rules():
            contains = await self._pos.check_allergen(allergen, severe)
        return _facts(
            allergen=allergen,
            contains=contains,
            sharedKitchen=True,
            next="Say which items contain it. Everything is made in one kitchen, so never call "
            "anything safe or free of it.",
        )

    @function_tool
    async def set_pickup(self, ctx: RunContext) -> str:
        """The caller will pick the order up."""
        with _pos_rules():
            await self._pos.set_fulfillment("pickup")
        return _facts(pickup=True, **self._money_facts(), **self._read_back_status())

    @function_tool
    async def set_delivery(
        self,
        ctx: RunContext,
        address: str | None = None,
        zip_code: str | None = None,
        driver_note: str | None = None,
    ) -> str:
        """Delivery: where to, and anything the driver should know. Pass only what they just
        told you, never a placeholder like "N/A"; what you leave out stays as it is.

        Args:
            address: The street address to deliver to, as they said it.
            zip_code: The five-digit ZIP code, only if they gave one.
            driver_note: Drop-off details for the driver ("leave it at the door"), in a few
                words, only if they gave some. It replaces the last note.
        """
        with _pos_rules():
            f = await self._pos.set_fulfillment(
                "delivery", address, zip_code, driver_note
            )
        return _facts(
            deliveryTo=f.address,
            driverNote=f.driver_note,
            **self._money_facts(),
            **self._read_back_status(),
        )

    @function_tool
    async def set_contact(
        self, ctx: RunContext, name: str | None = None, phone: str | None = None
    ) -> str:
        """The name for the order and a callback number. Pass what they just gave you.

        Args:
            name: The name for the order.
            phone: The callback number, all ten digits, written without spaces.
        """
        with _pos_rules():
            await self._pos.set_contact(name, phone)
        return _facts(name=self._pos.name, phone=self._pos.phone)

    @function_tool
    async def place_order(self, ctx: RunContext) -> str:
        """Send the order to the kitchen. The first call checks it's ready and returns what to
        read back; call it again once the caller has said it's right. After any change it's
        read back again."""
        if self._answered_review(ctx, "place"):
            with _pos_rules():
                order = await self._pos.place()
            self._must_tell(ctx, "their order number and when it'll be ready")
            return _facts(
                orderNumber=order.number,
                readyIn=order.ready_in,
                next="Tell them the order number and when it'll be ready, then ask if there's "
                "anything else.",
            )
        with _pos_rules():
            read_back = self._pos.order_read_back()
        self._read_back(ctx, "place")
        return _facts(
            placed=False,
            readBack=read_back,
            next="It isn't placed: they haven't heard this read-back yet. Read it to them now "
            "as a question (the items, how it gets to them, any note for the driver, and the "
            "total with what it includes) and wait for their answer. Call place_order again "
            "only once they agree.",
        )

    def _unfinished(self) -> str | None:
        if self._pos.order is None and self._pos.lines:
            return (
                "their order isn't placed. Ask whether to place it; if they want none of it, "
                "cancel_order."
            )
        return None


class DeliveryUpdate(Gabby):
    """Gabby calling the customer whose delivery is about to go out: something in it ran out,
    and anything else they'd like to change before it leaves."""

    def __init__(self, pos: Pizzeria) -> None:
        super().__init__(pos, instructions=_update_instructions(pos))

    async def on_enter(self) -> None:
        self.session.say(delivery_opener(self._pos))

    @function_tool
    async def swap_sold_out_item(self, ctx: RunContext, item_id: ItemId) -> str:
        """Replace the item that ran out with the one they choose, at the same price. Also when
        they change their mind, even after taking it off.

        Args:
            item_id: The replacement's menu id.
        """
        with _pos_rules():
            line = await self._pos.swap_sold_out_item(item_id)
        return _facts(nowOnTheOrder=line.label, samePrice=True, **self._money_facts())

    @function_tool
    async def take_off_sold_out_item(self, ctx: RunContext) -> str:
        """Take the item that ran out off the order, once they've said they'd rather not have a
        replacement."""
        with _pos_rules():
            less = await self._pos.take_off_sold_out_item()
        return _facts(lessAtTheDoor=money(less), **self._money_facts())

    @function_tool
    async def set_breadsticks(self, ctx: RunContext, orders: int) -> str:
        """How many orders of breadsticks to add, the only thing that can still go on the order,
        since they're ready to go. The total they want, not how many more; 0 takes them off.

        Args:
            orders: The number of orders of breadsticks in all.
        """
        with _pos_rules():
            change = await self._pos.set_breadsticks(orders)
        return _facts(
            breadsticks=orders,
            moreAtTheDoor=money(change) if change >= 0 else None,
            lessAtTheDoor=money(-change) if change < 0 else None,
            **self._money_facts(),
        )

    @function_tool
    async def set_driver_note(self, ctx: RunContext, note: str) -> str:
        """Save drop-off details for the driver: a gate code, the unit, leave it at the door. It
        replaces the last note. It can't change the address.

        Args:
            note: The details, in a few words.
        """
        with _pos_rules():
            await self._pos.set_driver_note(note)
        return _facts(driverNote=self._pos.fulfillment.driver_note)

    @function_tool
    async def read_back_changes(self, ctx: RunContext) -> str:
        """What changed on this call, what's due at the door, and when the driver gets there.
        Call it once the swap and the note for the driver are settled, and again after any later
        change."""
        with _pos_rules():
            facts = await self._pos.read_back_changes()
        self._must_tell(
            ctx, "what changed, what's due at the door, and when the driver gets there"
        )
        return _facts(
            **facts,
            next="Say each change once (not the whole order), with any code or unit number "
            'in a note digit by digit ("seven three oh nine") so they can catch a mishearing; '
            "what's due at the door, and when the driver will be there, with how that compares "
            "to before. If they've already said that's all, that's your goodbye: end_call in "
            "this same reply. Otherwise ask if there's anything else.",
        )

    @function_tool
    async def end_without_update(self, ctx: RunContext, reason: EndedEarly) -> str:
        """End the call without going through the update: a wrong number or someone else, a bad
        time, or they don't want to. The order goes on as it is.

        Args:
            reason: wrong_person, call_back_later, or declined.
        """
        with _pos_rules():
            await self._pos.end_early(reason)
        self._must_hear.clear()
        if reason == "wrong_person":
            return _facts(
                next="Apologize for the mix-up and say goodbye. Say nothing about the order, "
                "the address, or who it's for. Then end_call."
            )
        return _facts(next="The order goes on as it is. Say goodbye, then end_call.")

    def _money_facts(self) -> dict[str, Any]:
        return {
            "dueAtTheDoor": money(self._pos.due_now()),
            "comparedToBefore": compared(self._pos.balance()),
        }

    def _unfinished(self) -> str | None:
        pos = self._pos
        d = pos.delivery
        assert d is not None
        if d.ended_early or (pos.order and pos.order.cancelled):
            return None
        if pos.sold_out_unresolved():
            swaps = " or ".join(pos.same_price_swaps())
            return f"ask whether they'd like a {swaps} instead of the {pos.ran_out_line().label}, or to take it off."
        if d.read_back_revision != pos.revision:
            return "call read_back_changes and tell them what changed and what's due."
        return None


def delivery_opener(pos: Pizzeria) -> str:
    d = pos.delivery
    assert d is not None
    first = pos.name.split()[0] if pos.name else None
    hi = f"Hi {first}," if first else "Hi,"
    return (
        f"{hi} this is {PERSONA} from {RESTAURANT}, calling about your delivery order. "
        f"{d.driver} was packing it up to head out and realized there's no "
        f"{pos.ran_out_line().item.name} left. Would a "
        f"{pos.same_price_swaps()[0]} at the same price work instead, or should I take it off?"
    )


def _menu() -> str:
    lines = []
    for m in MENU.values():
        prices = ", ".join(
            f"{size_words(s)} {money(p)}".strip() for s, p in m.prices.items()
        )
        made_with = f"; {', '.join(m.ingredients)}" if m.ingredients else ""
        details = f"; {m.details}" if m.details else ""
        sold_out = " (sold out today)" if m.id in SOLD_OUT_TODAY else ""
        lines.append(
            f"- {m.id}: {m.name} {m.category}, {prices}{made_with}{details}{sold_out}"
        )
    return "\n".join(lines)


def _order_instructions() -> str:
    pizza_sizes = "; ".join(f"{size}, {about}" for size, about in PIZZA_SIZES.items())
    drink_sizes = ", ".join(f"{size} {about}" for size, about in DRINK_SIZES.items())
    toppings = ", ".join(t.name for t in TOPPINGS.values())
    sauces = ", ".join(s.name for s in SAUCES.values())
    return f"""\
You're {PERSONA}, answering the phone at {RESTAURANT}, {NEIGHBORHOOD}. Warm, quick, and lightly upbeat, like someone good at the counter on a busy night; calm and plain about allergies, complaints, or cancelling.
We're open {HOURS} Anything else about the restaurant that isn't here, like the address or parking, you don't know.

{VOICE_RULES}

The menu (ids are for your tools):
{_menu()}
Pizza sizes: {pizza_sizes}. Crust: {CRUST}. Drink sizes: {drink_sizes}.
Extra toppings on a pizza, $2 each, $1 on half: {toppings}. Wing sauces: {sauces}.

Take the order the way they give it:
- Put what they order on the order right away with add_items, everything they just named in one call, and ask only for what's missing, like a size or a wing sauce. A size they gave for one item is never another item's size.
- Never choose for them. For something we don't have, like a sauce, a crust, or a topping, say so and offer what we do have. Wings go on without a sauce until they pick one of ours, and "no sauce" is plain.
- An item they name again that's already on the order is the same item, not another one. A change is to one line: update_item or remove_item.
- Half-and-half is one pizza, a cheese pizza unless they name another, with first_half and second_half. "No olives" goes in without, at the same price.
- Allergies: check_allergen for each one, then say which items the menu lists it in.
- Once, when the order sounds complete and has no breadsticks, offer breadsticks. That's the only thing you suggest.

Then:
1. Pickup or delivery: set_pickup, or set_delivery. We deliver anywhere with a street address, for $4.99. Anything for the driver goes in driver_note.
2. A name for the order, then a callback number: set_contact.
3. place_order, without summing up the order yourself first. It returns the order to read back: read it as a question. When they agree, call place_order again, then give them the order number and when it'll be ready, and ask if there's anything else. Anything short of agreeing (a change, a question, "almost") gets sorted out and read back again first.
Orders under $15 have a $2.99 small-order fee: say it once in the read-back, and never push them to add items.
To cancel: call cancel_order before asking anything, confirm with them what it returns, and call it again once they agree."""


def _update_instructions(pos: Pizzeria) -> str:
    d = pos.delivery
    assert d is not None and pos.order is not None
    items = "; ".join(ln.describe() for ln in d.ordered.values())
    to = pos.name or "the customer, whose name you don't have: never say or guess one"
    return f"""\
You're {PERSONA} from {RESTAURANT}, calling {to} about their delivery order, number {pos.order.number}: {items}. It's going to {pos.fulfillment.address}. {d.driver}, the driver, is about to head out with it, about {d.eta_minutes} minutes to their door, and {money(d.due)} is due at the door, cash or card. Packing it up, {d.driver} found the {pos.ran_out_line().item.name} ran out after the order went in.
Polite, clear, and brief; calm and plain about money, a complaint, or cancelling.

{VOICE_RULES}

You've already said who you are and why you're calling, that {d.driver} is about to head out, and that the {pos.ran_out_line().item.name} ran out, and asked whether a {pos.same_price_swaps()[0]} at the same price works instead or to take it off. Whoever picked up is the customer: never ask who they are.
- Their choice: swap_sold_out_item (same-price options: {", ".join(pos.same_price_swaps())}), or take_off_sold_out_item. They can change their mind.
- Then ask once if there's anything {d.driver} should know, like a gate code or the unit: set_driver_note. The address can't change; if it needs to go somewhere else, they can cancel.
- Only breadsticks can be added, since they're ready to go: set_breadsticks. Nothing else can go on without holding {d.driver} up.
- A complaint: apologize sincerely, and offer only what you can do: the swap, taking it off, or cancelling. No discounts, credits, refunds, or free items.
- Cancelling the whole order: call cancel_order before asking anything, confirm with them what it returns, and call it again once they agree. The order doesn't go out and nothing is due.
- A wrong number, a bad time, or they'd rather not: end_without_update, before sharing anything more.
- Once the swap and the note for {d.driver} are settled: read_back_changes, say what it gives you, and ask if there's anything else. After any later change, read_back_changes again. Once they have nothing else, goodbye and end_call."""


def caller_script(pos: Pizzeria) -> str:
    """Who the caller is and what they want right now, for the suggested replies."""
    return _update_script(pos) if pos.delivery else _order_script(pos)


def _order_script(pos: Pizzeria) -> str:
    facts = """\
The caller is Sam Carter, ordering from Zappi's Pizzeria by phone. The only details to use:
- The order: a large pepperoni pizza and a dozen bone-in wings with hot sauce, for pickup. No breadsticks or anything else.
- Name: Sam Carter. Callback number: four one five, five five five, zero one four two."""
    if pos.order:
        stage = "The order is in. When Gabby asks if there's anything else, that's all, thanks, goodbye."
    elif pos.name and pos.phone:
        stage = "When Gabby reads the order back, yes, that's right."
    elif pos.fulfillment.mode:
        stage = "Give the name when Gabby asks for it, then the callback number."
    elif pos.lines:
        stage = "If Gabby asks about the wing sauce, it's hot. No breadsticks, thanks. That's everything, for pickup."
    else:
        stage = "Order the large pepperoni and the dozen hot wings, in one line."
    return f"{facts}\n{stage}"


def _update_script(pos: Pizzeria) -> str:
    facts = """\
Gabby from Zappi's Pizzeria called the caller about their delivery order, which is about to go out. The only details to use:
- A large Coke at the same price is fine instead of the Sprite that ran out.
- The driver should know the gate code is four five one two.
- Nothing else to add."""
    if pos.sold_out_unresolved():
        stage = "When Gabby asks about the Sprite, a Coke is fine."
    elif not pos.fulfillment.driver_note:
        stage = "When Gabby asks about anything for the driver, the gate code is four five one two."
    else:
        stage = "That's all. When Gabby reads back the changes and what's due, thanks, goodbye."
    return f"{facts}\n{stage}"
