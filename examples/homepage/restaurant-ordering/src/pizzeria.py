"""Zappi's Pizzeria's point of sale: the menu, an order being taken over the phone, and order #214,
already out with the driver.

No LiveKit in here. Everything the agent passes in is a choice from the menu (ids, sizes, sauces,
allergens) or a value the caller gave, so this checks structure and the restaurant's rules, never
what anyone meant. A request the restaurant won't take raises PizzeriaError, saying what to do
instead. The parts of the state a change touches are published together, through the callback
it's given."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal

DATA = json.loads((Path(__file__).parent / "zappis.json").read_text())

RESTAURANT: str = DATA["restaurant"]
NEIGHBORHOOD: str = DATA["neighborhood"]
HOURS: str = DATA["hours"]
PIZZA_SIZES: dict[str, str] = DATA["pizzaSizes"]
CRUST: str = DATA["crust"]
DRINK_SIZES: dict[str, str] = DATA["drinkSizes"]
TAX_RATE = Decimal(DATA["taxRate"])
DELIVERY_FEE = Decimal(DATA["deliveryFee"])
SMALL_ORDER_FEE = Decimal(DATA["smallOrderFee"])
SMALL_ORDER_UNDER = Decimal(DATA["smallOrderUnder"])
TOPPING_PRICE = Decimal(DATA["toppingPrice"])
MOST_OF_ONE_ITEM = 20
MOST_TIP = Decimal("100")
OUT = DATA["outForDelivery"]

Size = Literal["small", "medium", "large", "half_dozen", "dozen"]
Mode = Literal["pickup", "delivery"]
EndedEarly = Literal["wrong_person", "call_back_later", "declined"]
ItemId = Literal[tuple(m["id"] for m in DATA["menu"])]  # type: ignore[valid-type]
ToppingId = Literal[tuple(t["id"] for t in DATA["toppings"])]  # type: ignore[valid-type]
SauceId = Literal[tuple(s["id"] for s in DATA["sauces"])]  # type: ignore[valid-type]
Allergen = Literal[tuple(DATA["allergens"])]  # type: ignore[valid-type]
Ingredient = Literal[  # type: ignore[valid-type]
    tuple(sorted({i for m in DATA["menu"] for i in m.get("ingredients", [])}))
]

ENDED_EARLY: dict[str, str] = {
    "wrong_person": "Wrong number",
    "call_back_later": "Not a good time",
    "declined": "Didn't want to go through it",
}

Publish = Callable[[dict[str, Any]], Awaitable[None]]


async def _discard(_parts: dict[str, Any]) -> None:
    return None


class PizzeriaError(Exception):
    """A request the restaurant won't take, and what to do instead."""


@dataclass(frozen=True)
class MenuItem:
    id: str
    name: str
    category: str
    prices: dict[str, Decimal]  # by size; one-size items have the single size ""
    ingredients: tuple[str, ...]
    allergens: frozenset[str]
    details: str  # what a caller might ask about it: "6 to an order"

    @property
    def sizes(self) -> list[str]:
        return [s for s in self.prices if s]


@dataclass(frozen=True)
class Option:
    id: str
    name: str
    allergens: frozenset[str]


def _option(raw: dict[str, Any]) -> Option:
    return Option(raw["id"], raw["name"], frozenset(raw["allergens"]))


MENU: dict[str, MenuItem] = {
    m["id"]: MenuItem(
        id=m["id"],
        name=m["name"],
        category=m["category"],
        prices={k: Decimal(v) for k, v in m["prices"].items()}
        if "prices" in m
        else {"": Decimal(m["price"])},
        ingredients=tuple(m.get("ingredients", ())),
        allergens=frozenset(m["allergens"]),
        details=m.get("details", ""),
    )
    for m in DATA["menu"]
}
TOPPINGS: dict[str, Option] = {t["id"]: _option(t) for t in DATA["toppings"]}
SAUCES: dict[str, Option] = {s["id"]: _option(s) for s in DATA["sauces"]}
PLAIN = "plain"  # the wings' choice of no sauce at all
SOLD_OUT_TODAY = frozenset(m["id"] for m in DATA["menu"] if m.get("soldOut"))


def size_words(size: str | None) -> str:
    return (size or "").replace("_", " ")


def money(amount: Decimal) -> str:
    return f"${amount:,.2f}"


def _cents(amount: Decimal) -> Decimal:
    return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _or_list(words: Iterable[str]) -> str:
    words = list(words)
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"


@dataclass(frozen=True)
class Line:
    """One line of an order: an item in one configuration, and how many of it."""

    id: str
    item: MenuItem
    size: str | None
    quantity: int = 1
    toppings: tuple[Option, ...] = ()
    sauce: Option | None = None
    without: tuple[str, ...] = ()
    halves: tuple[tuple[Option, ...], tuple[Option, ...]] = ((), ())

    @property
    def unit_price(self) -> Decimal:
        on_halves = len(self.halves[0]) + len(self.halves[1])
        return (
            self.item.prices[self.size or ""]
            + TOPPING_PRICE * len(self.toppings)
            + TOPPING_PRICE / 2 * on_halves
        )

    @property
    def total(self) -> Decimal:
        return self.unit_price * self.quantity

    @property
    def allergens(self) -> frozenset[str]:
        options = (*self.toppings, *self.halves[0], *self.halves[1])
        if self.sauce:
            options += (self.sauce,)
        return self.item.allergens.union(*(o.allergens for o in options))

    @property
    def configuration(self) -> tuple[Any, ...]:
        return (
            self.item.id,
            self.size,
            frozenset(t.id for t in self.toppings),
            self.sauce.id if self.sauce else None,
            frozenset(self.without),
            frozenset(frozenset(t.id for t in h) for h in self.halves),
        )

    @property
    def label(self) -> str:
        """ "large Pepperoni", "Caesar Salad" """
        return f"{size_words(self.size)} {self.item.name}".strip()

    @property
    def needs(self) -> str | None:
        """What the caller hasn't chosen yet: wings stay on the order until they pick a sauce,
        so nothing they asked for drops off while it's sorted out."""
        if self.item.category == "wings" and self.sauce is None:
            return "sauce"
        return None

    @property
    def modifiers(self) -> list[str]:
        mods = []
        if self.sauce:
            mods.append(
                self.sauce.name
                if self.sauce.id == PLAIN
                else f"{self.sauce.name} sauce"
            )
        mods += [f"add {t.name}" for t in self.toppings]
        if any(self.halves):
            # which half is which doesn't matter
            first, second = sorted(
                ", ".join(t.name for t in h) or "plain" for h in self.halves
            )
            mods += [f"half {first}", f"other half {second}"]
        return mods + [f"no {i}" for i in self.without]

    def describe(self) -> str:
        """ "1 dozen Bone-in Wings, hot sauce" """
        mods = ", ".join(self.modifiers)
        return f"{self.quantity} {self.label}" + (f", {mods}" if mods else "")

    def facts(self) -> dict[str, Any]:
        return {
            "lineId": self.id,
            "quantity": self.quantity,
            "item": self.label,
            "with": self.modifiers,
            "price": money(self.total),
            **({"needs": self.needs} if self.needs else {}),
        }


@dataclass(frozen=True)
class Quote:
    subtotal: Decimal
    tax: Decimal
    delivery_fee: Decimal
    small_order_fee: Decimal
    tip: Decimal

    @property
    def total(self) -> Decimal:
        return (
            self.subtotal
            + self.tax
            + self.delivery_fee
            + self.small_order_fee
            + self.tip
        )

    def facts(self) -> dict[str, Any]:
        return {
            "subtotal": money(self.subtotal),
            "tax": money(self.tax),
            "deliveryFee": money(self.delivery_fee) if self.delivery_fee else None,
            "smallOrderFee": f"{money(self.small_order_fee)}, since it's under {money(SMALL_ORDER_UNDER)}"
            if self.small_order_fee
            else None,
            "tip": money(self.tip) if self.tip else None,
            "total": money(self.total),
        }


@dataclass
class Fulfillment:
    mode: Mode | None = None
    address: str | None = None
    zip_code: str | None = None
    driver_note: str | None = None


@dataclass
class Order:
    number: str
    ready_in: str
    cancelled: bool = False


@dataclass
class Delivery:
    """Order #214, already out with the driver: paid at the door, cash or card."""

    driver: str
    eta_minutes: int
    due: Decimal  # what was due at the door when it went out
    ordered: dict[str, Line]  # each line as it went out
    ran_out: str  # the line whose item ran out after it went in
    ended_early: EndedEarly | None = None
    read_back_revision: int | None = None


@dataclass
class Pizzeria:
    """One call's point of sale."""

    publish: Publish = _discard
    lines: list[Line] = field(default_factory=list)
    fulfillment: Fulfillment = field(default_factory=Fulfillment)
    name: str | None = None
    phone: str | None = None
    tip: Decimal = Decimal("0")
    allergies: dict[str, bool] = field(default_factory=dict)  # allergen -> severe
    sold_out: set[str] = field(default_factory=lambda: set(SOLD_OUT_TODAY))
    order: Order | None = None
    delivery: Delivery | None = None
    # bumped by every change to what's on the order, or how it gets to them: a read-back is
    # only good for the revision it read
    revision: int = 0
    _next_line: int = 1
    _next_order: int = DATA["firstOrderNumber"]

    @classmethod
    def out_for_delivery(
        cls, customer: str | None, publish: Publish = _discard
    ) -> Pizzeria:
        """Order #214: placed earlier, out with the driver now, and one item ran out after it
        went in. Under the customer's name if it's known; never a made-up one."""
        pos = cls(publish=publish, name=customer or None)
        for spec in OUT["items"]:
            pos.lines.append(
                pos._build(
                    pos._new_line_id(),
                    spec["itemId"],
                    spec["size"],
                    sauce=spec.get("sauce"),
                )
            )
        pos.fulfillment = Fulfillment("delivery", OUT["address"], OUT["zip"])
        pos.order = Order(OUT["orderNumber"], f"with {OUT['driver']}")
        quote = pos.quote()
        assert quote is not None
        ran_out = next(ln for ln in pos.lines if ln.item.id == OUT["ranOut"])
        pos.sold_out.add(ran_out.item.id)
        pos.delivery = Delivery(
            driver=OUT["driver"],
            eta_minutes=OUT["etaMinutes"],
            due=quote.total,
            ordered={ln.id: ln for ln in pos.lines},
            ran_out=ran_out.id,
        )
        return pos

    # ---- the menu ----

    def alternatives(self, item: MenuItem) -> list[str]:
        return [
            m.name
            for m in MENU.values()
            if m.category == item.category
            and m.id not in self.sold_out
            and m.id != item.id
        ]

    async def check_allergen(self, allergen: Allergen, severe: bool) -> list[str]:
        """What the menu lists the allergen in. Everything else only doesn't list it: the
        kitchen is shared."""
        self.allergies[allergen] = severe or self.allergies.get(allergen, False)
        await self.publish({"allergies": self._allergies_json()})
        contains = [m.name for m in MENU.values() if allergen in m.allergens]
        contains += [
            f"wings with {s.name} sauce"
            for s in SAUCES.values()
            if allergen in s.allergens
        ]
        contains += [
            f"{t.name} as a topping"
            for t in TOPPINGS.values()
            if allergen in t.allergens
        ]
        return contains

    # ---- taking an order ----

    async def add_items(
        self, specs: list[dict[str, Any]]
    ) -> tuple[list[Line], list[str]]:
        """Adds each item that can go on the order as asked; returns what went on and what
        couldn't, each with what to ask or do."""
        self._taking_order()
        added: list[Line] = []
        problems: list[str] = []
        for spec in specs:
            try:
                line = self._build(self._peek_line_id(), **spec)
            except PizzeriaError as e:
                problems.append(str(e))
                continue
            if waiting := self._waiting_on_sauce(line):
                problems.append(
                    f"{waiting.id}, the {waiting.label}, is waiting on its sauce. If this is that "
                    f"sauce, update_item {waiting.id} with it; only if they want another order "
                    f"besides {waiting.id}, add this once {waiting.id} has its sauce."
                )
                continue
            if same := self._matching(line, among=(*self.lines, *added)):
                problems.append(
                    f"{line.label} is already on the order as {same.id} ({same.describe()}). "
                    f"Saying it again isn't more: only if they ask for more, update_item "
                    f"{same.id} with the new total."
                )
                continue
            self._next_line += 1
            added.append(line)
        if added:
            self.lines += added
            await self._changed()
        return added, problems

    async def update_item(
        self,
        line_id: str,
        *,
        item_id: str | None = None,
        size: str | None = None,
        quantity: int | None = None,
        toppings: list[str] | None = None,
        sauce: str | None = None,
        without: list[str] | None = None,
        first_half: list[str] | None = None,
        second_half: list[str] | None = None,
    ) -> tuple[Line, bool]:
        """Changes one line: what isn't given stays as it is. Returns the line and whether
        anything changed."""
        self._taking_order()
        current = self._line(line_id)
        item = MENU[item_id] if item_id else current.item
        same_item = item.id == current.item.id
        keep_size = current.size if current.size in item.sizes else None
        line = self._build(
            current.id,
            item.id,
            size or keep_size,
            quantity if quantity is not None else current.quantity,
            toppings=toppings
            if toppings is not None
            else [t.id for t in current.toppings if item.category == "pizza"],
            sauce=sauce
            or (
                current.sauce.id if current.sauce and item.category == "wings" else None
            ),
            without=without
            if without is not None
            else [i for i in current.without if i in item.ingredients],
            first_half=first_half
            if first_half is not None
            else [t.id for t in current.halves[0] if item.category == "pizza"],
            second_half=second_half
            if second_half is not None
            else [t.id for t in current.halves[1] if item.category == "pizza"],
            sold_out_ok=same_item,
        )
        if line == current:
            return current, False
        others = [ln for ln in self.lines if ln.id != line.id]
        if same := self._matching(line, among=others):
            raise PizzeriaError(
                f"That would make it the same as {same.id} ({same.describe()}). Change "
                f"{same.id}'s quantity instead, and remove {line.id}."
            )
        self.lines = [line if ln.id == line.id else ln for ln in self.lines]
        await self._changed()
        return line, True

    async def remove_item(self, line_id: str) -> Line:
        self._taking_order()
        line = self._line(line_id)
        self.lines = [ln for ln in self.lines if ln.id != line.id]
        await self._changed()
        return line

    async def set_fulfillment(
        self,
        mode: Mode | None = None,
        address: str | None = None,
        zip_code: str | None = None,
        driver_note: str | None = None,
    ) -> Fulfillment:
        """Pickup or delivery. For delivery, what isn't given stays as it is."""
        self._taking_order()
        mode = mode or self.fulfillment.mode
        if mode is None:
            raise PizzeriaError("Ask whether it's pickup or delivery first.")
        if mode == "pickup":
            if driver_note:
                raise PizzeriaError(
                    "It's a pickup order, so there's no driver to tell. If they meant "
                    "delivery, ask for the address."
                )
            new = Fulfillment("pickup")
        else:
            was = (
                self.fulfillment
                if self.fulfillment.mode == "delivery"
                else Fulfillment()
            )
            new = Fulfillment(
                "delivery",
                _text(address) or was.address,
                _zip(zip_code) if zip_code else was.zip_code,
                _note(driver_note) if driver_note is not None else was.driver_note,
            )
            if not new.address:
                raise PizzeriaError("Ask for the street address to deliver to.")
        if new != self.fulfillment:
            self.fulfillment = new
            await self._changed()
        return new

    async def set_contact(
        self, name: str | None = None, phone: str | None = None
    ) -> None:
        self._taking_order()
        if phone is not None:
            digits = "".join(c for c in phone if c.isdigit())
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]
            if not _real_us_number(digits):
                raise PizzeriaError(
                    f"{digits or 'That'} isn't a ten-digit US number. If what they said is one, "
                    "pass all ten digits again; otherwise ask them to say it again."
                )
            self.phone = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        if name is not None and _text(name):
            self.name = _text(name)

    async def set_tip(
        self, dollars: float | None = None, percent: float | None = None
    ) -> Decimal:
        """A tip the caller asked for, in dollars or a percent of the subtotal; 0 takes it off."""
        self._can_change()
        if (dollars is None) == (percent is None):
            raise PizzeriaError(
                "Pass the tip as dollars or as a percent, one of the two."
            )
        if percent is not None:
            if not 0 <= percent <= 100:
                raise PizzeriaError("A tip percent is between 0 and 100.")
            amount = self.subtotal() * Decimal(str(percent)) / 100
        else:
            amount = Decimal(str(dollars))
        tip = _cents(amount)
        if not 0 <= tip <= MOST_TIP:
            raise PizzeriaError(f"A tip is between $0 and {money(MOST_TIP)}.")
        if tip != self.tip:
            self.tip = tip
            await self._changed()
        return tip

    def order_read_back(self) -> dict[str, Any]:
        """The order as it would go to the kitchen, if it's ready to go."""
        self._ready_to_place()
        quote = self.quote()
        assert quote is not None
        f = self.fulfillment
        return {
            "items": [ln.describe() for ln in self.lines],
            "pickupOrDelivery": "pickup"
            if f.mode == "pickup"
            else f"delivery to {f.address}",
            "driverNote": f.driver_note,
            **quote.facts(),
            "payment": self.payment(),
        }

    async def place(self) -> Order:
        self._ready_to_place()
        if self.fulfillment.mode == "delivery":
            lo, hi = DATA["deliveryMinutes"]
            ready_in = f"delivered in about {lo} to {hi} minutes"
        else:
            ready_in = f"ready for pickup in about {DATA['pickupMinutes']} minutes"
        self.order = Order(str(self._next_order), ready_in)
        self._next_order += 1
        await self.publish({"order": self._order_json()})
        return self.order

    def cancel_read_back(self) -> dict[str, Any]:
        order = self._cancellable()
        if self.delivery:
            return {
                "cancel": f"order {order.number}",
                "driverTurnsBack": self.delivery.driver,
                "due": money(Decimal("0")),
            }
        return {"cancel": f"order {order.number}", "due": money(Decimal("0"))}

    async def cancel(self) -> Order:
        order = self._cancellable()
        order.cancelled = True
        self.revision += 1
        await self.publish(
            {"order": self._order_json(), "delivery": self._delivery_json()}
        )
        return order

    def drop_read_back(self) -> dict[str, Any]:
        """What dropping an order that hasn't been placed takes off: all of it."""
        self._taking_order()
        if not self.lines:
            raise PizzeriaError("Nothing's on the order, so there's nothing to drop.")
        return {
            "drop": [ln.describe() for ln in self.lines],
            "placed": False,
            "due": money(Decimal("0")),
        }

    async def drop(self) -> None:
        """Takes everything off an order that hasn't been placed, its tip included."""
        self._taking_order()
        self.lines, self.tip = [], Decimal("0")
        await self._changed()

    # ---- the order out for delivery ----

    def ran_out_line(self) -> Line:
        d = self._out()
        return self.lines_by_id().get(d.ran_out) or d.ordered[d.ran_out]

    def same_price_swaps(self) -> list[str]:
        """What the item that ran out can be swapped for at the same price: "large Coke"."""
        was = self._out().ordered[self._out().ran_out]
        return [
            f"{size_words(size)} {m.name}".strip()
            for m in MENU.values()
            if m.category == was.item.category and m.id not in self.sold_out
            for size, price in m.prices.items()
            if price == was.item.prices[was.size or ""]
        ]

    def sold_out_unresolved(self) -> bool:
        line = self.lines_by_id().get(self._out().ran_out)
        return line is not None and line.item.id in self.sold_out

    async def swap_sold_out_item(self, item_id: str) -> Line:
        """Swaps the item that ran out for another of the same kind at the same price, or puts
        it back on as that if it was taken off."""
        d = self._on_the_way()
        was = d.ordered[d.ran_out]
        item = MENU[item_id]
        if item.id in self.sold_out:
            raise PizzeriaError(
                f"The {item.name} is out too. Same-price options: {_or_list(self.same_price_swaps())}."
            )
        size = next(
            (s for s, p in item.prices.items() if p == was.item.prices[was.size or ""]),
            None,
        )
        if item.category != was.item.category or size is None:
            raise PizzeriaError(
                f"The {was.label} can only be swapped for something at the same price: "
                f"{_or_list(self.same_price_swaps())}. Or take it off."
            )
        line = replace(was, item=item, size=size or None)
        current = self.lines_by_id().get(d.ran_out)
        if current == line:
            return line
        self.lines = sorted(
            [ln for ln in self.lines if ln.id != d.ran_out] + [line],
            key=lambda ln: int(ln.id[1:]),
        )
        await self._changed()
        return line

    async def take_off_sold_out_item(self) -> Decimal:
        """Takes the item that ran out off the order; returns how much less is due at the door."""
        d = self._on_the_way()
        if d.ran_out not in self.lines_by_id():
            raise PizzeriaError(
                f"The {d.ordered[d.ran_out].label} is already off the order."
            )
        before = self.due_now()
        self.lines = [ln for ln in self.lines if ln.id != d.ran_out]
        await self._changed()
        return before - self.due_now()

    async def set_breadsticks(self, orders: int) -> Decimal:
        """How many orders of breadsticks go with the driver, who carries spares; returns the
        change in what's due at the door."""
        d = self._on_the_way()
        carried = OUT["driverCarries"][0]
        if not 0 <= orders <= OUT["mostExtras"]:
            raise PizzeriaError(
                f"{d.driver} carries at most {OUT['mostExtras']} orders of breadsticks."
            )
        before = self.due_now()
        existing = next((ln for ln in self.lines if ln.item.id == carried), None)
        if (existing.quantity if existing else 0) == orders:
            return Decimal("0")
        others = [ln for ln in self.lines if ln.item.id != carried]
        if orders:
            line = (
                replace(existing, quantity=orders)
                if existing
                else self._build(self._new_line_id(), carried, None, orders)
            )
            others.append(line)
        self.lines = others
        await self._changed()
        return self.due_now() - before

    async def set_driver_note(self, note: str) -> str:
        """Drop-off details for the driver of the order on its way. Replaces the last note."""
        self._on_the_way()
        new = replace(self.fulfillment, driver_note=_note(note))
        if new != self.fulfillment:
            self.fulfillment = new
            await self._changed()
        return note

    def changes(self) -> list[str]:
        """What changed on this call, against the order as it went out."""
        d = self._out()
        if self.order and self.order.cancelled:
            return [f"Order {self.order.number} cancelled: nothing is due"]
        now = self.lines_by_id()
        out: list[str] = []
        for line_id, was in d.ordered.items():
            line = now.get(line_id)
            if line is None:
                out.append(f"{was.label} taken off")
            elif line.item.id != was.item.id:
                out.append(f"{was.label} swapped for a {line.label}, same price")
        out += [
            f"{ln.quantity} {'order' if ln.quantity == 1 else 'orders'} of {ln.label} added"
            for line_id, ln in now.items()
            if line_id not in d.ordered
        ]
        if self.fulfillment.driver_note:
            out.append(f"Note for {d.driver}: {self.fulfillment.driver_note}")
        if self.tip:
            out.append(f"{money(self.tip)} tip for {d.driver}")
        return out

    async def read_back_changes(self) -> dict[str, Any]:
        """The final read-back of the call about the order on its way: what changed, what's
        due at the door, and when the driver gets there."""
        d = self._out()
        if self.sold_out_unresolved() and not (self.order and self.order.cancelled):
            line = self.ran_out_line()
            raise PizzeriaError(
                f"The {line.label} is still on the order. Ask whether they'd like "
                f"{_or_list(self.same_price_swaps())} instead, or to take it off."
            )
        d.read_back_revision = self.revision
        await self.publish({"delivery": self._delivery_json()})
        cancelled = bool(self.order and self.order.cancelled)
        return {
            "changes": self.changes() or ["no changes"],
            "dueAtTheDoor": money(self.due_now()),
            "comparedToBefore": compared(self.balance()),
            "driver": d.driver,
            "arrivesInMinutes": None if cancelled else d.eta_minutes,
        }

    async def end_early(self, reason: EndedEarly) -> None:
        d = self._out()
        d.ended_early = reason
        await self.publish({"delivery": self._delivery_json()})

    def due_now(self) -> Decimal:
        if self.order and self.order.cancelled:
            return Decimal("0")
        quote = self.quote()
        return quote.total if quote else Decimal("0")

    def balance(self) -> Decimal:
        """How much more (+) or less (-) is due at the door than when the order went out."""
        return self.due_now() - self._out().due

    # ---- reading it ----

    def lines_by_id(self) -> dict[str, Line]:
        return {ln.id: ln for ln in self.lines}

    def subtotal(self) -> Decimal:
        return sum((ln.total for ln in self.lines), Decimal("0"))

    def quote(self) -> Quote | None:
        if not self.lines:
            return None
        subtotal = self.subtotal()
        return Quote(
            subtotal=subtotal,
            tax=_cents(subtotal * TAX_RATE),
            delivery_fee=DELIVERY_FEE
            if self.fulfillment.mode == "delivery"
            else Decimal("0"),
            small_order_fee=SMALL_ORDER_FEE
            if subtotal < SMALL_ORDER_UNDER
            else Decimal("0"),
            tip=self.tip,
        )

    def payment(self) -> str | None:
        if self.order and self.order.cancelled:
            return "Nothing is due: the order is cancelled"
        if self.fulfillment.mode == "pickup":
            return "Pay at pickup, cash or card"
        if self.fulfillment.mode == "delivery":
            return "Pay the driver at the door, cash or card"
        return None

    def still_to_ask(self) -> list[str]:
        """What the caller still has to choose, line by line, before it can be read back."""
        return [
            f"Which sauce on {ln.id}, the {ln.label}: "
            f"{_or_list(s.name for s in SAUCES.values())}. They choose; if they'd rather not "
            f"have them, remove_item {ln.id}."
            for ln in self.lines
            if ln.needs == "sauce"
        ]

    def cart_facts(self) -> dict[str, Any]:
        quote = self.quote()
        asks = self.still_to_ask()
        return {
            "order": [ln.facts() for ln in self.lines],
            **({"subtotal": money(quote.subtotal)} if quote else {}),
            **({"stillToAsk": asks} if asks else {}),
        }

    async def publish_all(self) -> None:
        await self.publish(
            {
                "menu": self._menu_json(),
                "allergies": self._allergies_json(),
                "order": self._order_json(),
                **self._order_state_json(),
            }
        )

    # ---- inside ----

    def _build(
        self,
        line_id: str,
        item_id: str,
        size: str | None = None,
        quantity: int = 1,
        *,
        toppings: list[str] | None = None,
        sauce: str | None = None,
        without: list[str] | None = None,
        first_half: list[str] | None = None,
        second_half: list[str] | None = None,
        sold_out_ok: bool = False,
    ) -> Line:
        item = MENU[item_id]
        if item.id in self.sold_out and not sold_out_ok:
            raise PizzeriaError(
                f"The {item.name} is sold out today. Offer one or two of these instead: "
                f"{', '.join(self.alternatives(item))}."
            )
        if item.sizes:
            if size is None:
                raise PizzeriaError(
                    f"Ask which size {item.name}: {_or_list(size_words(s) for s in item.sizes)}."
                )
            if size not in item.sizes:
                raise PizzeriaError(
                    f"{item.name} comes in {_or_list(size_words(s) for s in item.sizes)}, "
                    f"not {size_words(size)}. Ask which."
                )
        else:
            size = None
        if not 1 <= quantity <= MOST_OF_ONE_ITEM:
            raise PizzeriaError(f"A quantity is between 1 and {MOST_OF_ONE_ITEM}.")
        is_pizza = item.category == "pizza"
        if (toppings or first_half or second_half or without) and not is_pizza:
            raise PizzeriaError(
                f"Only pizzas take toppings or changes, not the {item.name}."
            )
        if sauce and item.category != "wings":
            raise PizzeriaError(f"The {item.name} doesn't take a sauce choice.")
        whole = tuple(dict.fromkeys(TOPPINGS[t] for t in toppings or ()))
        halves = (
            tuple(dict.fromkeys(TOPPINGS[t] for t in first_half or ())),
            tuple(dict.fromkeys(TOPPINGS[t] for t in second_half or ())),
        )
        if both := [t.name for t in halves[0] if t in halves[1]]:
            raise PizzeriaError(
                f"{' and '.join(both)} on both halves goes on the whole pizza: pass it in "
                "toppings instead."
            )
        if doubled := [t.name for t in (*halves[0], *halves[1]) if t in whole]:
            raise PizzeriaError(
                f"{' and '.join(doubled)} is already on the whole pizza."
            )
        if missing := [i for i in without or () if i not in item.ingredients]:
            raise PizzeriaError(
                f"The {item.name} doesn't come with {' or '.join(missing)}, so there's nothing "
                f"to leave off. It comes with {', '.join(item.ingredients)}."
            )
        return Line(
            id=line_id,
            item=item,
            size=size,
            quantity=quantity,
            toppings=whole,
            sauce=SAUCES[sauce] if sauce else None,
            without=tuple(dict.fromkeys(without or ())),
            halves=halves,
        )

    def _matching(self, line: Line, among: Iterable[Line]) -> Line | None:
        return next(
            (ln for ln in among if ln.configuration == line.configuration), None
        )

    def _waiting_on_sauce(self, line: Line) -> Line | None:
        """Wings already on the order without a sauce, which this line names one for. Only lines
        from before: two orders of wings asked for at once are two orders."""
        if line.sauce is None:
            return None
        return next(
            (
                ln
                for ln in self.lines
                if ln.needs == "sauce"
                and (ln.item.id, ln.size) == (line.item.id, line.size)
            ),
            None,
        )

    def _line(self, line_id: str) -> Line:
        if line := self.lines_by_id().get(line_id):
            return line
        on_order = ", ".join(ln.id for ln in self.lines) or "none"
        raise PizzeriaError(
            f"There's no line {line_id} on the order. Lines: {on_order}."
        )

    def _peek_line_id(self) -> str:
        return f"L{self._next_line}"

    def _new_line_id(self) -> str:
        line_id = self._peek_line_id()
        self._next_line += 1
        return line_id

    def _taking_order(self) -> None:
        if self.delivery:
            raise PizzeriaError(
                f"Order {OUT['orderNumber']} is already on its way: only the item that ran out, "
                "breadsticks, a note for the driver, and a tip can change."
            )
        if self.order:
            state = "cancelled" if self.order.cancelled else "sent to the kitchen"
            raise PizzeriaError(f"Order {self.order.number} was already {state}.")

    def _can_change(self) -> None:
        if self.delivery:
            self._on_the_way()
        else:
            self._taking_order()

    def _ready_to_place(self) -> None:
        self._taking_order()
        if not self.lines:
            raise PizzeriaError("Nothing's on the order yet.")
        if asks := self.still_to_ask():
            raise PizzeriaError(" ".join(asks))
        if self.fulfillment.mode is None:
            raise PizzeriaError(
                "Pickup or delivery isn't set: set_pickup or set_delivery if they've said, or "
                "ask."
            )
        if not self.name:
            raise PizzeriaError(
                "There's no name on the order: set_contact if they gave one, or ask."
            )
        if not self.phone:
            raise PizzeriaError(
                "There's no callback number: set_contact if they gave one, or ask."
            )

    def _cancellable(self) -> Order:
        if self.delivery:
            self._on_the_way()
        if self.order is None:
            raise PizzeriaError(
                "Nothing has been placed, so there's nothing to cancel and nothing is due."
            )
        if self.order.cancelled:
            raise PizzeriaError(f"Order {self.order.number} is already cancelled.")
        return self.order

    def _out(self) -> Delivery:
        if self.delivery is None:
            raise PizzeriaError("There's no order out for delivery on this call.")
        return self.delivery

    def _on_the_way(self) -> Delivery:
        d = self._out()
        if d.ended_early == "wrong_person":
            raise PizzeriaError(
                "Whoever answered isn't the customer: share and change nothing. Say goodbye."
            )
        if self.order and self.order.cancelled:
            raise PizzeriaError(
                f"Order {self.order.number} is cancelled: {d.driver} is turning back and "
                "nothing is due."
            )
        return d

    async def _changed(self) -> None:
        self.revision += 1
        await self.publish(self._order_state_json())

    def _order_state_json(self) -> dict[str, Any]:
        return {
            "cart": self._cart_json(),
            "fulfillment": self._fulfillment_json(),
            "quote": self._quote_json(self.quote()),
            "delivery": self._delivery_json(),
        }

    def _delivery_json(self) -> dict[str, Any] | None:
        d = self.delivery
        if d is None:
            return None
        cancelled = bool(self.order and self.order.cancelled)
        return {
            "driver": d.driver,
            "etaMinutes": d.eta_minutes,
            "due": float(d.due),
            "dueNow": float(self.due_now()),
            "balance": float(self.balance()),
            "note": self.fulfillment.driver_note,
            "cancelled": cancelled,
            "changes": [] if cancelled else self.changes(),
            "endedEarly": ENDED_EARLY[d.ended_early] if d.ended_early else None,
            "readBack": d.read_back_revision == self.revision,
        }

    def _menu_json(self) -> list[dict[str, Any]]:
        return [
            {
                "id": m.id,
                "name": m.name,
                "category": m.category,
                "allergens": sorted(m.allergens),
                "available": m.id not in self.sold_out,
            }
            for m in MENU.values()
        ]

    def _cart_json(self) -> dict[str, Any]:
        ordered = self.delivery.ordered if self.delivery else {}
        return {
            "version": self.revision,
            "lines": [
                _line_json(ln, ordered.get(ln.id), self.sold_out) for ln in self.lines
            ],
            "subtotal": float(self.subtotal()),
        }

    def _fulfillment_json(self) -> dict[str, Any]:
        f = self.fulfillment
        return {
            "mode": f.mode,
            "address": f.address,
            "zip": f.zip_code,
            "note": f.driver_note,
        }

    def _quote_json(self, quote: Quote | None) -> dict[str, Any] | None:
        if quote is None:
            return None
        return {
            "subtotal": float(quote.subtotal),
            "tax": float(quote.tax),
            "deliveryFee": float(quote.delivery_fee),
            "smallOrderFee": float(quote.small_order_fee),
            "tip": float(quote.tip),
            "total": float(quote.total),
        }

    def _order_json(self) -> dict[str, Any] | None:
        o = self.order
        if o is None:
            return None
        return {"number": o.number, "readyIn": o.ready_in, "cancelled": o.cancelled}

    def _allergies_json(self) -> list[dict[str, Any]]:
        return [{"allergen": a, "severe": s} for a, s in self.allergies.items()]


def _line_json(line: Line, was: Line | None, sold_out: set[str]) -> dict[str, Any]:
    # the page shows wings by their count ("12 Bone-in Wings"), and a swapped line's original
    pieces = {"half_dozen": "6", "dozen": "12"}
    swapped = was is not None and was.item.id != line.item.id
    return {
        "id": line.id,
        "itemId": line.item.id,
        "name": line.item.name,
        "size": pieces.get(line.size or "", size_words(line.size)) or None,
        "quantity": line.quantity,
        "modifiers": line.modifiers,
        "needs": line.needs,
        "total": float(line.total),
        "out": line.item.id in sold_out,
        "was": _capitalized(was.label) if was and swapped else None,
    }


def _capitalized(label: str) -> str:
    return label[:1].upper() + label[1:]


def compared(balance: Decimal) -> str:
    if balance > 0:
        return f"{money(balance)} more than before"
    if balance < 0:
        return f"{money(-balance)} less than before"
    return "the same as before"


def _text(value: str | None) -> str:
    return " ".join((value or "").split())


def _zip(value: str) -> str:
    digits = "".join(c for c in value if c.isdigit())
    if len(digits) != 5:
        raise PizzeriaError(
            "A ZIP code is five digits. Ask for it again, or leave it out."
        )
    return digits


def _note(value: str) -> str:
    note = _text(value)
    if not note:
        raise PizzeriaError("The note is empty. Ask what the driver should know.")
    if len(note) > 200:
        raise PizzeriaError(
            "That's too long for the driver. Ask for the key details in a few words."
        )
    return note


def _real_us_number(digits: str) -> bool:
    if len(digits) != 10 or len(set(digits)) == 1:
        return False
    area, exchange = digits[:3], digits[3:6]
    return all(part[0] not in "01" and part[1:] != "11" for part in (area, exchange))
