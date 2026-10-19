"""A mock point-of-sale for Zappi's Pizzeria: the menu, the cart, the order it submits, and
(on the outbound call) a delivery order that's already out with the driver."""

from __future__ import annotations

import base64
import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

Category = Literal["pizza", "wings", "sides", "dips", "salads", "drinks", "desserts"]
Mode = Literal["pickup", "delivery"]
EndedEarly = Literal["wrong_person", "call_back_later", "declined"]

CATEGORIES: tuple[Category, ...] = (
    "pizza",
    "wings",
    "sides",
    "dips",
    "salads",
    "drinks",
    "desserts",
)
DELIVERY_FEE = Decimal("4.99")
# a small order (pickup or delivery) pays a flat fee, untaxed like the delivery fee
SMALL_ORDER_FEE = Decimal("2.99")
SMALL_ORDER_UNDER = Decimal("15.00")
TAX_RATE = Decimal("0.09")
UPSELL_ITEM = "breadsticks"
MAX_QUANTITY = 20
PICKUP_MINUTES = 20
DELIVERY_MINUTES = (40, 50)
CENT = Decimal("0.01")
# the delivery-update call: order #214 is out with Marco, and the Sprite ran out after it went in
DRIVER = "Marco"
ETA_MINUTES = 5
DRIVER_EXTRAS = ("breadsticks",)  # the driver carries spare breadsticks; nothing else can be added
MAX_EXTRAS = 2
NOTE_MAX = 200
MAX_TIP = Decimal("100.00")
ENDED_EARLY: dict[EndedEarly, str] = {
    "wrong_person": "Wrong person",
    "call_back_later": "Asked to call back later",
    "declined": "Declined the update",
}


class PosError(Exception):
    """A request the POS won't take, phrased so the agent can relay it."""


def real_us_number(digits: str) -> bool:
    """Ten digits that could be a real North American number: an area code and exchange that
    don't start with 0 or 1 and aren't N11 codes (411, 911), and not one digit repeated."""
    if not re.fullmatch(r"\d{10}", digits) or len(set(digits)) == 1:
        return False
    area, exchange = digits[:3], digits[3:6]
    return all(p[0] not in "01" and p[1:] != "11" for p in (area, exchange))


def _key(s: str) -> str:
    """Lowercase, no accents, no punctuation: "Jalapeños" and "jalapenos" are the same."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def _cents(d: Decimal) -> Decimal:
    return d.quantize(CENT, rounding=ROUND_HALF_UP)


def _singular(w: str) -> str:
    return w[:-2] if w.endswith("oes") else w[:-1] if w.endswith("s") else w


def _loose(s: str, filler: frozenset[str]) -> str:
    """_key without filler words or plurals, so "a large", "hot sauce", "cokes" and "meat
    lover's pizza" match "large", "hot", "coke" and "Meat Lovers"."""
    words = (_singular(w) for w in _key(s).split() if w not in filler)
    return " ".join(w for w in words if w)


SIZE_FILLER = frozenset({"a", "an", "the", "size", "sized", "piece", "pieces", "pc", "count"})
OPTION_FILLER = frozenset({"a", "an", "the", "some", "extra", "add", "with", "sauce", "topping"})
ITEM_FILLER = frozenset({"a", "an", "the", "order", "of", "and", "pizza", "pie"})


def _words(s: str) -> set[str]:
    # singular words, minus the "no" / "hold the" around them: "hold the onions" -> {"onion"}
    out = set()
    for w in _key(s).split():
        if w not in ("no", "without", "hold", "the", "any"):
            out.add(w[:-2] if w.endswith("oes") else w[:-1] if w.endswith("s") else w)
    return out


@dataclass(frozen=True)
class Size:
    id: str
    label: str  # how it's said: "large", "a dozen"
    price: Decimal
    aliases: tuple[str, ...] = ()
    pieces: int | None = None  # wings are sized by the count: a dozen is one order of 12

    def matches(self, said: str) -> bool:
        k = _key(said)
        if k == _key(self.id) or k == _key(self.label) or k in self.aliases:
            return True
        names = (self.id, self.label, *self.aliases)
        loose = _loose(said, SIZE_FILLER)
        return bool(loose) and loose in {_loose(n, SIZE_FILLER) for n in names}


@dataclass(frozen=True)
class Option:
    id: str
    label: str
    price: Decimal = Decimal("0")
    allergens: frozenset[str] = frozenset()
    aliases: tuple[str, ...] = ()

    def matches(self, said: str) -> bool:
        k = _key(said)
        if k in (_key(self.id), _key(self.label), *self.aliases):
            return True
        names = (self.id, self.label, *self.aliases)
        loose = _loose(said, OPTION_FILLER)
        return bool(loose) and loose in {_loose(n, OPTION_FILLER) for n in names}


@dataclass
class MenuItem:
    id: str
    name: str
    category: Category
    sizes: tuple[Size, ...]
    allergens: frozenset[str] = frozenset()
    blurb: str = ""
    available: bool = True
    ingredients: tuple[str, ...] = ()  # what a pizza comes with, so it can be left off

    @property
    def ref(self) -> str:
        # the id the model sees on the menu and passes back to the tools: opaque, so the only way
        # to an item is the name beside its id, never a word the caller said or a category
        digest = hashlib.sha256(self.id.encode()).digest()
        return base64.b32encode(digest)[:4].decode().lower()

    @property
    def needs_size(self) -> bool:
        return len(self.sizes) > 1

    @property
    def takes_toppings(self) -> bool:
        return self.category == "pizza"

    @property
    def needs_sauce(self) -> bool:
        return self.category == "wings"

    def size(self, said: str | None) -> Size:
        if not self.needs_size:
            return self.sizes[0]
        labels = [s.label for s in self.sizes]
        # "small, medium, or large", the way Gabby asks it
        options = (
            " or ".join(labels) if len(labels) < 3 else f"{', '.join(labels[:-1])}, or {labels[-1]}"
        )
        if not said:
            raise PosError(f"{self.name} needs a size: {options}. Ask which, naming each one.")
        for s in self.sizes:
            if s.matches(said):
                return s
        raise PosError(f"{self.name} comes in {options}, not {said!r}. Ask which, naming each one.")

    def describe(self) -> str:
        if self.needs_size:
            prices = ", ".join(f"{s.label} ${s.price}" for s in self.sizes)
        else:
            prices = f"${self.sizes[0].price}"
        extra = " Sauce required." if self.needs_sauce else ""
        return f"{self.ref}: {self.name} ({prices}).{extra}"


# ---- the menu ----

NUTS = frozenset({"nuts"})
DAIRY_GLUTEN = frozenset({"dairy", "gluten"})


def _pizza(
    id: str, name: str, small: str, blurb: str, ingredients: tuple[str, ...], **kw: Any
) -> MenuItem:
    s = Decimal(small)
    return MenuItem(
        id=id,
        name=name,
        category="pizza",
        sizes=(
            Size("S", "small", s, ("sm", "10", "10 inch", "personal")),
            Size("M", "medium", s + 3, ("med", "12", "12 inch")),
            Size("L", "large", s + 6, ("lg", "14", "14 inch", "big")),
        ),
        allergens=kw.pop("allergens", DAIRY_GLUTEN),
        blurb=blurb,
        ingredients=ingredients,
        **kw,
    )


RED = ("tomato sauce", "mozzarella cheese")


def _wings(id: str, name: str, six: str, twelve: str) -> MenuItem:
    return MenuItem(
        id=id,
        name=name,
        category="wings",
        sizes=(
            Size("6", "half dozen", Decimal(six), ("half a dozen", "six", "a half dozen"), 6),
            Size("12", "dozen", Decimal(twelve), ("a dozen", "twelve", "full dozen"), 12),
        ),
        allergens=frozenset({"dairy"}),
    )


def _soda(id: str, name: str) -> MenuItem:
    return MenuItem(
        id=id,
        name=name,
        category="drinks",
        sizes=(
            Size("S", "small", Decimal("1.99"), ("sm",)),
            Size("M", "medium", Decimal("2.99"), ("med", "regular")),
            Size("L", "large", Decimal("3.99"), ("lg", "big")),
        ),
    )


def _one(id: str, name: str, category: Category, price: str, **kw: Any) -> MenuItem:
    return MenuItem(
        id=id, name=name, category=category, sizes=(Size("one", "", Decimal(price)),), **kw
    )


def seed_menu() -> list[MenuItem]:
    return [
        _pizza("cheese", "Cheese", "11.99", "Tomato, mozzarella", RED),
        _pizza("margherita", "Margherita", "12.99", "Tomato, mozzarella, basil", (*RED, "basil")),
        _pizza("pepperoni", "Pepperoni", "13.99", "Cup-and-char pepperoni", (*RED, "pepperoni")),
        _pizza(
            "sausage_peppers",
            "Sausage & Peppers",
            "14.99",
            "Fennel sausage, bell peppers",
            (*RED, "fennel sausage", "bell peppers"),
        ),
        _pizza(
            "veggie",
            "Garden Veggie",
            "13.99",
            "Mushroom, onion, pepper, olive",
            (*RED, "mushrooms", "onions", "green peppers", "black olives"),
        ),
        _pizza(
            "pesto",
            "Pesto Chicken",
            "14.99",
            "Basil pine-nut pesto, chicken",
            ("pesto", "mozzarella cheese", "chicken"),
            allergens=DAIRY_GLUTEN | NUTS,
        ),
        _pizza(
            "white",
            "White Pie",
            "13.99",
            "Ricotta, garlic, mozzarella",
            ("ricotta", "garlic", "mozzarella cheese"),
        ),
        _pizza(
            "meat_lovers",
            "Meat Lovers",
            "15.99",
            "Pepperoni, sausage, bacon",
            (*RED, "pepperoni", "sausage", "bacon"),
        ),
        _pizza(
            "bbq_chicken",
            "BBQ Chicken",
            "14.99",
            "BBQ sauce, chicken, red onion",
            ("bbq sauce", "mozzarella cheese", "chicken", "red onions"),
        ),
        _pizza("hawaiian", "Hawaiian", "14.99", "Ham, pineapple", (*RED, "ham", "pineapple")),
        _wings("wings", "Bone-in Wings", "8.99", "16.99"),
        _wings("boneless", "Boneless Wings", "7.99", "14.99"),
        _one("breadsticks", "Breadsticks", "sides", "5.99", allergens=DAIRY_GLUTEN, blurb="Six"),
        _one(
            "buffalo_cauliflower",
            "Buffalo Cauliflower",
            "sides",
            "8.99",
            allergens=frozenset({"dairy"}),
            available=False,  # 86'd today
        ),
        _one(
            "mozz_sticks",
            "Mozzarella Sticks",
            "sides",
            "7.99",
            allergens=DAIRY_GLUTEN | {"egg"},
        ),
        _one("fries", "Fries", "sides", "4.49"),
        _one("cheesy_bread", "Cheesy Bread", "sides", "6.99", allergens=DAIRY_GLUTEN),
        # dipping sauces, for the crusts, breadsticks, and wings
        _one("garlic_sauce", "Garlic Sauce", "dips", "0.99", allergens=frozenset({"dairy"})),
        _one("ranch", "Ranch", "dips", "0.99", allergens=frozenset({"dairy", "egg"})),
        _one("marinara", "Marinara", "dips", "0.99"),
        _one("blue_cheese", "Blue Cheese", "dips", "0.99", allergens=frozenset({"dairy", "egg"})),
        _one(
            "caesar",
            "Caesar Salad",
            "salads",
            "9.49",
            allergens=DAIRY_GLUTEN | {"egg", "fish"},
        ),
        _one("house_salad", "House Salad", "salads", "7.99"),
        _one("caprese", "Caprese Salad", "salads", "10.49", allergens=frozenset({"dairy"})),
        _soda("coke", "Coke"),
        _soda("diet_coke", "Diet Coke"),
        _soda("sprite", "Sprite"),
        MenuItem(
            id="lemonade",
            name="Lemonade",
            category="drinks",
            sizes=(
                Size("reg", "regular", Decimal("2.99"), ("small", "16 oz")),
                Size("lg", "large", Decimal("3.99"), ("big", "32 oz")),
            ),
        ),
        _one("water", "Bottled Water", "drinks", "1.79"),
        _one("cannoli", "Cannoli", "desserts", "4.49", allergens=DAIRY_GLUTEN),
        _one(
            "tiramisu",
            "Tiramisu",
            "desserts",
            "6.49",
            allergens=DAIRY_GLUTEN | {"egg"},
        ),
    ]


TOPPINGS = (
    Option("extra_cheese", "extra cheese", Decimal("2.00"), aliases=("cheese", "double cheese")),
    Option("mushrooms", "mushrooms", Decimal("2.00"), aliases=("mushroom",)),
    Option("onions", "onions", Decimal("2.00"), aliases=("onion", "red onion", "red onions")),
    Option("peppers", "green peppers", Decimal("2.00"), aliases=("pepper", "bell peppers")),
    Option("olives", "black olives", Decimal("2.00"), aliases=("olive", "olives")),
    Option("jalapenos", "jalapeños", Decimal("2.00"), aliases=("jalapeno", "jalapenos")),
    Option("pepperoni", "pepperoni", Decimal("2.00"), aliases=("extra pepperoni",)),
    Option("sausage", "sausage", Decimal("2.00")),
    Option("bacon", "bacon", Decimal("2.00")),
    Option("pineapple", "pineapple", Decimal("2.00")),
)

SAUCES = (
    Option("hot", "hot", aliases=("buffalo", "hot buffalo", "spicy")),
    Option("mild", "mild", aliases=("mild buffalo",)),
    Option("bbq", "BBQ", aliases=("barbecue", "bbq")),
    Option(
        "garlic_parm", "garlic parmesan", allergens=frozenset({"dairy"}), aliases=("garlic parm",)
    ),
    Option("lemon_pepper", "lemon pepper"),
    Option("thai", "Thai chili peanut", allergens=NUTS, aliases=("thai", "thai chili", "peanut")),
)

DIGIT_WORDS = {
    w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine".split())
} | {"oh": "0", "o": "0"}

# what a caller says -> the allergen tag in the menu data
ALLERGEN_WORDS = {
    "nuts": ("nut", "nuts", "peanut", "peanuts", "tree nut", "tree nuts", "pine nut", "pine nuts"),
    "dairy": ("dairy", "milk", "lactose", "cheese"),
    "gluten": ("gluten", "wheat", "celiac", "coeliac"),
    "egg": ("egg", "eggs"),
    "fish": ("fish", "anchovy", "anchovies", "seafood"),
}


def _option(options: tuple[Option, ...], said: str, what: str) -> Option:
    for o in options:
        if o.matches(said):
            return o
    names = ", ".join(o.label for o in options)
    raise PosError(
        f"There's no {what} called {said!r}. If they meant one of these, retry with it; if not, "
        f"say we don't have it and name one or two close ones, not the whole list: {names}."
    )


# ---- the cart ----


def _halves_key(halves: tuple[list[Option], list[Option]]) -> frozenset[frozenset[str]]:
    # which half is which doesn't matter: half mushrooms, half pepperoni either way round
    return frozenset(frozenset(t.id for t in h) for h in halves)


@dataclass
class Line:
    id: str
    item: MenuItem
    size: Size
    quantity: int
    toppings: list[Option] = field(default_factory=list)
    sauce: Option | None = None
    without: list[str] = field(default_factory=list)  # ingredients left off: "black olives"
    # half-and-half: each half's own toppings, at half a topping's price
    halves: tuple[list[Option], list[Option]] = field(default_factory=lambda: ([], []))
    rev: int = 1  # bumped on every edit, so the page knows which line to flash
    was: str | None = None  # the line before its last edit, for the page to strike through

    @property
    def unit_price(self) -> Decimal:
        half = sum((t.price / 2 for t in (*self.halves[0], *self.halves[1])), Decimal("0"))
        return self.size.price + sum((t.price for t in self.toppings), Decimal("0")) + half

    @property
    def total(self) -> Decimal:
        return self.unit_price * self.quantity

    @property
    def allergens(self) -> frozenset[str]:
        opts = (
            *self.toppings,
            *self.halves[0],
            *self.halves[1],
            *([self.sauce] if self.sauce else []),
        )
        extra = [o.allergens for o in opts]
        return self.item.allergens.union(*extra)

    def same_order_as(self, other: Line) -> bool:
        return (
            self.item.id == other.item.id
            and self.size.id == other.size.id
            and {t.id for t in self.toppings} == {t.id for t in other.toppings}
            and (self.sauce and self.sauce.id) == (other.sauce and other.sauce.id)
            and set(self.without) == set(other.without)
            and _halves_key(self.halves) == _halves_key(other.halves)
        )

    def modifiers(self) -> list[str]:
        mods = [f"{self.sauce.label} sauce"] if self.sauce else []
        mods += [f"+ {t.label}" for t in self.toppings]
        if any(self.halves):
            first, second = (", ".join(t.label for t in h) or "plain" for h in self.halves)
            mods += [f"half {first}", f"other half {second}"]
        return mods + [f"no {w}" for w in self.without]

    def describe(self) -> str:
        size = f"{self.size.label} " if self.item.needs_size else ""
        mods = ", ".join(self.modifiers())
        return f"{self.id}: {self.quantity} × {size}{self.item.name}" + (
            f", {mods}" if mods else ""
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "itemId": self.item.id,
            "name": self.item.name,
            # wings show their count ("12 Bone-in Wings"), though "a dozen" is how it's said
            "size": (
                (str(self.size.pieces) if self.size.pieces else self.size.label)
                if self.item.needs_size
                else None
            ),
            "quantity": self.quantity,
            "modifiers": self.modifiers(),
            "unitPrice": float(self.unit_price),
            "total": float(self.total),
            "allergens": sorted(self.allergens),
            "rev": self.rev,
            "was": self.was,
            "out": not self.item.available,
        }


@dataclass
class Quote:
    version: int  # the cart version this total was worked out (and read back) for
    subtotal: Decimal
    tax: Decimal
    delivery_fee: Decimal
    small_order_fee: Decimal
    tip: Decimal
    total: Decimal

    def describe(self) -> str:
        fee = f", delivery ${self.delivery_fee}" if self.delivery_fee else ""
        small = (
            f", small-order fee ${self.small_order_fee} (it's under ${SMALL_ORDER_UNDER})"
            if self.small_order_fee
            else ""
        )
        tip = f", tip ${self.tip}" if self.tip else ""
        return f"subtotal ${self.subtotal}, tax ${self.tax}{fee}{small}{tip}, total ${self.total}"


@dataclass
class Fulfillment:
    mode: Mode | None = None
    address: str | None = None
    zip: str | None = None
    note: str | None = None  # drop-off details for the driver, on a delivery


@dataclass
class Order:
    number: str
    ready_in: str
    note: str | None = None  # the drop-off details sent with it, for a delivery
    cancelled: bool = False  # cancelled on this call: nothing is due


@dataclass
class Delivery:
    """An order already out for delivery: what's due at the door, and what changed on the
    call."""

    driver: str
    eta_minutes: int
    due: Decimal  # what was due at the door (cash or card) when the order went out
    note: str | None = None  # drop-off details for the driver
    changes: dict[str, str] = field(default_factory=dict)  # kind -> what changed, for read-back
    rev: int = 0  # bumped on every change, so a read-back can be told apart from a stale one
    read_back_rev: int | None = None
    # line id -> the line as ordered, for an out-of-stock line swapped or taken off on this call
    ordered: dict[str, Line] = field(default_factory=dict)
    ended_early: EndedEarly | None = None  # the call ended without going through the update


@dataclass
class Pos:
    menu: list[MenuItem]
    lines: list[Line] = field(default_factory=list)
    removed: list[Line] = field(default_factory=list)
    fulfillment: Fulfillment = field(default_factory=Fulfillment)
    name: str | None = None
    phone: str | None = None
    allergies: dict[str, bool] = field(default_factory=dict)  # allergen -> severe
    upsell_offered: bool = False
    tip: Decimal = Decimal("0")  # in dollars, on top of the total; paid with it in person
    # the cancellation was read back to the caller ("Cancel the whole order?"), and nothing has
    # changed since: only then can it go through
    cancel_asked: bool = False
    # every change to what's being bought, or how it's handed over, bumps the version
    version: int = 0
    quote: Quote | None = None
    order: Order | None = None
    last_touched: str | None = None  # the line added or changed most recently
    delivery: Delivery | None = None  # set only for an order that's already out
    _next_line: int = 1

    # ---- the menu ----

    def item(self, item_id: str) -> MenuItem:
        k = _key(item_id)
        for m in self.menu:
            if k in (_key(m.ref), _key(m.id), _key(m.name)):
                return m
        loose = _loose(item_id, ITEM_FILLER)
        for m in self.menu:
            if loose and loose in (_loose(m.id, ITEM_FILLER), _loose(m.name, ITEM_FILLER)):
                return m
        raise PosError(
            f"{item_id!r} isn't on the menu. If they meant a menu item, retry with its id; if "
            "not, say we don't have it and suggest something close from the menu."
        )

    def substitutes(self, item: MenuItem) -> list[MenuItem]:
        return [m for m in self.menu if m.category == item.category and m.available and m != item]

    def check_availability(self, item_id: str) -> MenuItem:
        item = self.item(item_id)
        if not item.available:
            subs = ", ".join(m.name for m in self.substitutes(item))
            raise PosError(
                f"The {item.name} is sold out today, so it can't be ordered. Tell the caller and "
                f"offer one or two of these instead: {subs}."
            )
        return item

    def with_allergen(self, said: str) -> tuple[str, list[str]]:
        # returns the allergen tag and the items (or item + sauce) that list it; everything
        # else only *doesn't list* it, since the kitchen is shared. The tag can be anywhere in
        # what they said: "gluten-free", "lactose intolerant", "peanut allergy"
        k = _key(said)
        tags = [
            t
            for t, words in ALLERGEN_WORDS.items()
            if any(re.search(rf"\b{w}\b", k) for w in (t, *words))
        ]
        if len(tags) > 1:
            raise PosError(f"That's {' and '.join(tags)}: look up one allergen per call.")
        if not tags:
            raise PosError(
                f"The menu data doesn't track {said!r}, so you can't tell them either way. Say "
                "so plainly, and that everything's made in one shared kitchen, so with a serious "
                "allergy they may want to skip anything they're unsure of. You handle this "
                "yourself: don't offer to pass them to anyone."
            )
        tag = tags[0]
        found = [m.name for m in self.menu if tag in m.allergens]
        found += [f"wings with {s.label} sauce" for s in SAUCES if tag in s.allergens]
        return tag, found

    def note_allergy(self, allergen: str, severe: bool) -> None:
        # every allergy mentioned stays noted; severe once said, severe for the call
        self.allergies[allergen] = severe or self.allergies.get(allergen, False)

    # ---- the cart ----

    def _editable(self) -> None:
        if self.order and self.order.cancelled:
            raise PosError(f"Order {self.order.number} was cancelled on this call.")
        if self.order:
            raise PosError(f"Order {self.order.number} was already sent to the kitchen.")

    def _touch(self) -> None:
        self.version += 1
        self.cancel_asked = False

    def line(self, line_id: str) -> Line:
        for ln in self.lines:
            if ln.id.lower() == line_id.strip().lower():
                return ln
        ids = ", ".join(ln.id for ln in self.lines) or "none"
        raise PosError(f"There's no line {line_id!r} in the cart. Lines: {ids}.")

    def _modifiers(
        self, item: MenuItem, toppings: list[str] | None, sauce: str | None
    ) -> tuple[list[Option], Option | None]:
        tops: list[Option] = []
        for t in toppings or []:
            if not item.takes_toppings:
                raise PosError(f"{item.name} doesn't take toppings.")
            if re.match(r"(no|without|hold)\b", _key(t)):
                raise PosError(
                    f"{t!r} is something to leave off: pass it in without, not toppings."
                )
            opt = _option(TOPPINGS, t, "topping")
            if opt not in tops:
                tops.append(opt)
        if item.needs_sauce:
            if not sauce:
                names = ", ".join(s.label for s in SAUCES)
                raise PosError(f"{item.name} need a sauce: {names}. Ask which one.")
            return tops, _option(SAUCES, sauce, "sauce")
        if sauce:
            raise PosError(f"{item.name} doesn't come with a sauce choice.")
        return tops, None

    def _halves(
        self,
        item: MenuItem,
        first: list[str] | None,
        second: list[str] | None,
        tops: list[Option],
    ) -> tuple[list[Option], list[Option]]:
        # "half mushroom, half pepperoni": one pizza, each half with its own toppings
        if not first and not second:
            return [], []
        if not item.takes_toppings:
            raise PosError(f"{item.name} doesn't take toppings, so it can't be half-and-half.")
        halves: list[list[Option]] = []
        for said in (first or [], second or []):
            half: list[Option] = []
            for t in said:
                opt = _option(TOPPINGS, t, "topping")
                if opt in tops:
                    raise PosError(
                        f"{opt.label} is already on the whole pizza: take it out of toppings, "
                        "or leave it off the half."
                    )
                if opt not in half:
                    half.append(opt)
            halves.append(half)
        if both := [t.label for t in halves[0] if t in halves[1]]:
            raise PosError(
                f"{' and '.join(both)} on both halves is a topping on the whole pizza: put it in "
                "toppings instead."
            )
        return halves[0], halves[1]

    def _left_off(self, item: MenuItem, without: list[str], tops: list[Option]) -> list[str]:
        # "no olives": only what that pizza comes with, as the menu names it
        out: list[str] = []
        for said in without:
            if not item.takes_toppings:
                raise PosError(
                    f"Only pizzas take changes, so nothing can be left off the {item.name}. If "
                    f"{said!r} was meant for a pizza in the order, put it on that pizza's line. "
                    f"Otherwise say the {item.name} comes as it's made, without saying what's "
                    "in it: the menu doesn't list that."
                )
            words = _words(said)
            ing = next((i for i in item.ingredients if words and words <= _words(i)), None)
            if ing is None:
                raise PosError(
                    f"The {item.name} doesn't come with {said!r} (it's "
                    f"{', '.join(item.ingredients)}), so there's nothing to leave off: tell them "
                    "it doesn't have that anyway. Ask only if they seem to mean something else."
                )
            if any(t.matches(ing) for t in tops):
                raise PosError(f"{ing} is an added topping on this line: take it out of toppings.")
            if ing not in out:
                out.append(ing)
        return out

    def add(
        self,
        item_id: str,
        quantity: int = 1,
        size: str | None = None,
        toppings: list[str] | None = None,
        sauce: str | None = None,
        without: list[str] | None = None,
        first_half: list[str] | None = None,
        second_half: list[str] | None = None,
    ) -> Line:
        self._editable()
        item = self.check_availability(item_id)
        _check_quantity(quantity)
        tops, sc = self._modifiers(item, toppings, sauce)
        halves = self._halves(item, first_half, second_half, tops)
        left_off = self._left_off(item, without or [], tops)
        chosen = item.size(size)
        _check_pieces(chosen, quantity)
        line = Line(f"L{self._next_line}", item, chosen, quantity, tops, sc, left_off, halves)
        # the same item said twice is one line, never two; changing how many is an explicit update
        for existing in self.lines:
            if existing.same_order_as(line):
                raise PosError(
                    f"That's already in the cart as {existing.describe()}: the caller is "
                    "repeating themselves, so leave it as it is and don't add to the count. "
                    'Only if they asked for more ("another one", "make it three", "put the '
                    'other one back", or yes to your offer of one more) use update_item on '
                    f"{existing.id} with the new total."
                )
        self._next_line += 1
        self.lines.append(line)
        self.last_touched = line.id
        self._touch()
        return line

    def update(
        self,
        line_id: str,
        quantity: int | None = None,
        size: str | None = None,
        toppings: list[str] | None = None,
        sauce: str | None = None,
        without: list[str] | None = None,
        more: bool = False,
        first_half: list[str] | None = None,
        second_half: list[str] | None = None,
        item_id: str | None = None,
    ) -> Line:
        # only the fields given change; raising a count needs `more`, since a repeated
        # "two pepperonis" is the most common way an order doubles by mistake. `item_id`
        # changes what the line is ("make the Coke a Sprite"), keeping its count, and its size
        # and modifiers where the new item takes them
        self._editable()
        line = self.line(line_id)
        if quantity is not None and quantity > line.quantity and not more:
            raise PosError(
                f"{line.id} already has {line.quantity}, and the caller didn't ask for more: "
                "saying the item again is repeating it, so the count stays. If you're not "
                "sure, ask how many they want in total."
            )
        before = line.describe()
        item = self.check_availability(item_id) if item_id else line.item
        swapped = item.id != line.item.id
        if size:
            new_size = item.size(size)
        elif not swapped:
            new_size = line.size
        else:
            # the size it had, if the new item comes in it: a large Coke becomes a large Sprite
            kept = (s for s in item.sizes if s.matches(line.size.label) or s.matches(line.size.id))
            new_size = item.sizes[0] if not item.needs_size else next(kept, None) or item.size(None)
        new_tops, new_sauce = line.toppings, line.sauce
        if toppings is not None or sauce is not None or swapped:
            old_tops = [t.label for t in line.toppings] if item.takes_toppings else None
            old_sauce = line.sauce.label if line.sauce and item.needs_sauce else None
            new_tops, new_sauce = self._modifiers(
                item,
                toppings if toppings is not None else old_tops,
                sauce if sauce is not None else old_sauce,
            )
        new_halves = line.halves
        if first_half is not None or second_half is not None or toppings is not None or swapped:
            halves = line.halves if item.takes_toppings else ([], [])
            new_halves = self._halves(
                item,
                first_half if first_half is not None else [t.label for t in halves[0]],
                second_half if second_half is not None else [t.label for t in halves[1]],
                new_tops,
            )
        # what's left off is about one item's ingredients, so a new item starts without any
        old_without = [] if swapped else line.without
        new_without = self._left_off(
            item, without if without is not None else old_without, new_tops
        )
        if quantity is not None:
            _check_quantity(quantity)
        _check_pieces(new_size, quantity if quantity is not None else line.quantity)
        changed = (
            swapped
            or new_size != line.size
            or new_tops != line.toppings
            or new_sauce != line.sauce
            or new_without != line.without
            or new_halves != line.halves
            or (quantity is not None and quantity != line.quantity)
        )
        if not changed:
            return line  # a repeat of what's already there: nothing to do
        line.item, line.size, line.toppings, line.sauce, line.without, line.halves = (
            item,
            new_size,
            new_tops,
            new_sauce,
            new_without,
            new_halves,
        )
        if quantity is not None:
            line.quantity = quantity
        line.was = before.split(": ", 1)[1]
        line.rev += 1
        self.last_touched = line.id
        self._touch()
        return line

    def remove(self, line_id: str) -> Line:
        self._editable()
        line = self.line(line_id)
        self.lines.remove(line)
        self.removed.append(line)
        self._touch()
        return line

    def offer_upsell(self) -> MenuItem:
        # the one upsell, once
        self._editable()
        breadsticks = self.item(UPSELL_ITEM)
        if self.upsell_offered:
            raise PosError("Breadsticks were already offered on this call. Don't offer again.")
        if any(ln.item.id == UPSELL_ITEM for ln in self.lines):
            raise PosError("Breadsticks are already in the order. No upsell needed.")
        self.upsell_offered = True
        return breadsticks

    # ---- pickup or delivery, and who it's for ----

    def set_pickup(self) -> None:
        self._editable()
        if self.fulfillment.mode != "pickup":
            self.fulfillment = Fulfillment(mode="pickup")
            self._touch()

    def set_delivery(self, address: str) -> None:
        self._editable()
        address = " ".join(address.split())
        # anywhere goes, but the driver needs a street address: a number and a name, not just a
        # ZIP or a city
        if not re.search(r"\d", address) or not re.search(r"[^\W\d_]{2,}", address):
            raise PosError("That isn't a full street address. Ask for the street and number.")
        m = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
        # a corrected address keeps the drop-off details already given
        note = self.fulfillment.note
        self.fulfillment = Fulfillment("delivery", address, m and m.group(1), note)
        self._touch()

    def set_delivery_note(self, note: str) -> str:
        # drop-off details for the driver ("leave it at the door"), read back with the order
        self._editable()
        if self.fulfillment.mode != "delivery":
            raise PosError(
                "It's a pickup order, so there's no driver to pass a note to. Say so plainly; "
                "if they meant delivery, ask for the address."
            )
        note = _driver_note(note)
        if looks_like_an_address(note):
            raise PosError(
                "That's an address, not a note for the driver. If they want it delivered there, "
                "call set_delivery with it; the note is for things like leave it at the door."
            )
        if note != self.fulfillment.note:
            self.fulfillment.note = note
            self._touch()
        return note

    def set_contact(self, name: str | None = None, phone: str | None = None) -> None:
        self._editable()
        if phone is not None:
            # "four one five, five five five, oh one four two" is a number too
            digits = "".join(
                DIGIT_WORDS.get(w)
                or (w.replace("o", "0") if re.fullmatch(r"[\do]+", w) else re.sub(r"\D", "", w))
                for w in re.findall(r"\w+", phone.lower())
            )
            if len(digits) == 11 and digits.startswith("1"):
                digits = digits[1:]
            if len(digits) != 10 or not real_us_number(digits):
                raise PosError(
                    "That callback number isn't a real ten-digit number. Ask them to say it "
                    'again ("Sorry, what was that number?"), without mentioning digits; never '
                    "fill in or guess digits."
                )
            self.phone = f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        if name is not None and name.strip():
            self.name = " ".join(name.split()).title()

    # ---- pricing and submitting ----

    def subtotal(self) -> Decimal:
        return sum((ln.total for ln in self.lines), Decimal("0"))

    def price(self) -> Quote:
        if not self.lines:
            raise PosError("The cart is empty.")
        if self.fulfillment.mode is None:
            raise PosError("Ask pickup or delivery first; delivery changes the total.")
        self.quote = self._quote(self.subtotal())
        return self.quote

    def _quote(self, subtotal: Decimal) -> Quote:
        fee = DELIVERY_FEE if self.fulfillment.mode == "delivery" else Decimal("0")
        small = SMALL_ORDER_FEE if subtotal < SMALL_ORDER_UNDER else Decimal("0")
        tax = _cents(subtotal * TAX_RATE)
        total = subtotal + tax + fee + small + self.tip
        return Quote(self.version, subtotal, tax, fee, small, self.tip, total)

    @property
    def quote_current(self) -> bool:
        return self.quote is not None and self.quote.version == self.version

    def submit(self) -> Order:
        self._editable()
        if not self.lines:
            raise PosError("The cart is empty.")
        if self.fulfillment.mode is None:
            raise PosError("Pickup or delivery hasn't been chosen.")
        if not self.name or not self.phone:
            raise PosError("Get the caller's name and a callback number first.")
        if self.quote is None:
            raise PosError("Price the order and read it back with the total before submitting.")
        if not self.quote_current:
            raise PosError(
                "The order changed after it was read back. Price it again, tell the caller the "
                "change and the new total, and get their okay."
            )
        if self.fulfillment.mode == "delivery":
            lo, hi = DELIVERY_MINUTES
            ready = f"delivered in about {lo} to {hi} minutes"
        else:
            ready = f"ready for pickup in about {PICKUP_MINUTES} minutes"
        number = f"{random.randint(100, 999)}"
        self.order = Order(number, ready, self.fulfillment.note)
        return self.order

    def set_tip(self, amount: Decimal | None = None, percent: Decimal | None = None) -> Decimal:
        """Sets the tip, in dollars or as a percent of the subtotal (0 takes it off); returns
        it. On the delivery-update call it's for the driver, added to what's due at the door."""
        if (amount is None) == (percent is None):
            raise PosError("Pass the tip as a dollar amount or a percent, one of the two.")
        if self.delivery:
            self._open()
        else:
            self._editable()
        if percent is not None:
            if not self.lines:
                raise PosError("The cart is empty, so there's nothing to work a percent from.")
            if not 0 <= percent <= 100:
                raise PosError("A tip percent has to be between 0 and 100.")
            amount = self.subtotal() * percent / 100
        assert amount is not None
        tip = _cents(amount)
        if not 0 <= tip <= MAX_TIP:
            raise PosError(f"A tip has to be between $0 and ${MAX_TIP}.")
        if tip == self.tip:
            return tip
        self.tip = tip
        self._touch()
        if self.delivery:
            self.price()
            d = self.delivery
            self._changed("tip", f"Tip for {d.driver} ${tip}, added at the door" if tip else None)
        return tip

    def payment(self) -> str | None:
        """How the order is paid, as the caller hears it: always in person, never over the
        phone."""
        if self.order and self.order.cancelled:
            return "Nothing due: the order was cancelled"
        if self.fulfillment.mode == "pickup":
            return "Pay at pickup, cash or card"
        if self.fulfillment.mode == "delivery":
            return "Pay the driver at the door, cash or card"
        return None

    def cancel(self, confirmed: bool = False) -> Literal["not_placed", "ask", "cancelled"]:
        """Cancels the order placed on this call (or, on the delivery-update call, the one out
        with the driver). It's read back first: "ask" means tell the caller what cancelling
        means and wait for a clear yes; only a `confirmed` call after that cancels it. Before
        an order is placed there's nothing to cancel: "not_placed"."""
        if self.delivery:
            self._open()
        elif self.order is None:
            return "not_placed"
        assert self.order is not None
        if self.order.cancelled:
            raise PosError(f"Order {self.order.number} is already cancelled. Nothing is due.")
        if not confirmed or not self.cancel_asked:
            self.cancel_asked = True
            return "ask"
        self.order.cancelled = True
        self.cancel_asked = False
        if d := self.delivery:
            # the cancellation replaces whatever else changed on the call
            d.changes.clear()
            self._changed("cancel", f"Cancelled order {self.order.number}; nothing is due")
        return "cancelled"

    # ---- an order out for delivery ----

    def _en_route(self) -> Delivery:
        if self.delivery is None:
            raise PosError("There's no order out for delivery on this call.")
        return self.delivery

    def _live(self) -> Delivery:
        # the order out for delivery, as long as it's the customer on the line
        d = self._en_route()
        if d.ended_early == "wrong_person":
            raise PosError(
                "Whoever answered isn't the customer: share nothing about the order, the address, "
                "or who it's for. Apologize, say goodbye, and call end_call."
            )
        return d

    def _open(self) -> Delivery:
        # the order out for delivery, with the customer on the line, and not cancelled
        d = self._live()
        if self.order and self.order.cancelled:
            raise PosError(
                f"Order {self.order.number} was cancelled on this call, so it can't change: "
                f"{d.driver} is turning back and nothing is due."
            )
        return d

    def _changed(self, kind: str, what: str | None) -> None:
        # what=None takes that change back off the read-back
        d = self._en_route()
        if what is None:
            d.changes.pop(kind, None)
        else:
            d.changes[kind] = what
        d.rev += 1
        self.cancel_asked = False
        d.ended_early = None  # they changed something, so they're going through the update

    def out_of_stock(self) -> list[Line]:
        return [ln for ln in self.lines if not ln.item.available]

    def _out_line(self, line_id: str) -> tuple[Line, Line]:
        # an out-of-stock line, or one already swapped or taken off on this call (the customer
        # can change their mind); returns it and the line as ordered
        d = self._open()
        key = line_id.strip().upper()
        if key in d.ordered:
            line = next(ln for ln in (*self.lines, *self.removed) if ln.id == key)
            return line, d.ordered[key]
        line = self.line(line_id)
        if line.item.available:
            raise PosError(
                f"{line.item.name} isn't out of stock. The rest of the order is already with "
                "the driver and can't be changed."
            )
        d.ordered[line.id] = replace(line)
        return line, d.ordered[line.id]

    def _taken_off(self, line: Line) -> bool:
        return all(ln.id != line.id for ln in self.lines)

    def same_price_swaps(self, line: Line) -> list[str]:
        # in stock, same category, a size at the same price: "large Coke"
        out = []
        for m in self.substitutes(line.item):
            for s in m.sizes:
                if s.price == line.size.price:
                    out.append(f"{s.label} {m.name}".strip() if m.needs_size else m.name)
        return out

    def due_now(self) -> Decimal:
        # what the customer pays the driver at the door: nothing once it's cancelled
        d = self._en_route()
        if self.order and self.order.cancelled:
            return Decimal("0")
        return self.quote.total if self.quote else d.due

    def balance(self) -> Decimal:
        # how much more (+) or less (-) is due at the door than when the order went out
        return self.due_now() - self._en_route().due

    def swap_out_of_stock(self, line_id: str, item_id: str) -> Line:
        line, ordered = self._out_line(line_id)
        new = self.check_availability(item_id)
        swaps = ", ".join(self.same_price_swaps(ordered)) or "none"
        if new.category != ordered.item.category:
            raise PosError(
                f"Swap the {ordered.item.name} only for another {ordered.item.category} item at "
                f"the same price: {swaps}. Or take it off the order."
            )
        size = next((s for s in new.sizes if s.price == ordered.size.price), None)
        if size is None:
            raise PosError(
                f"{new.name} doesn't come at ${ordered.size.price}. Same-price swaps: {swaps}. Or "
                "take it off the order."
            )
        if self._taken_off(line):
            # taken off earlier on this call: it goes back on the order, at its price
            self.removed.remove(line)
            self.lines.append(line)
            self.lines.sort(key=lambda ln: int(ln.id[1:]))
        elif line.item.id == new.id:
            return line  # already the one they picked
        before = line.describe().split(": ", 1)[1]
        line.item, line.size = new, size
        line.was, line.rev = before, line.rev + 1
        self.last_touched = line.id
        self._touch()
        self.price()
        was = ordered.describe().split(": ", 1)[1]
        self._changed("swap", f"{was} → {line.describe().split(': ', 1)[1]}, same price")
        return line

    def remove_out_of_stock(self, line_id: str) -> tuple[Line, Decimal]:
        # returns the line and how much less is due at the door
        line, ordered = self._out_line(line_id)
        if self._taken_off(line):
            raise PosError(f"The {ordered.item.name} was already taken off the order.")
        before = self.quote.total if self.quote else Decimal("0")
        self.lines.remove(line)
        if line.item.id != ordered.item.id:
            # swapped earlier on this call: it comes off as what was ordered
            line.was, line.rev = line.describe().split(": ", 1)[1], line.rev + 1
            line.item, line.size = ordered.item, ordered.size
        self.removed.append(line)
        self._touch()
        less = before - self.price().total
        self._changed(
            "swap", f"Removed {line.describe().split(': ', 1)[1]}, ${less} less at the door"
        )
        return line, less

    def add_extra(self, item_id: str, quantity: int = 1) -> tuple[Line, Decimal]:
        # adds the item, or sets how many there are (0 takes them off); returns the line and
        # the change in what's due at the door
        d = self._open()
        item = self.item(item_id)
        if item.id not in DRIVER_EXTRAS:
            raise PosError(
                f"{item.name} can't be added now: {d.driver} already left with the order. Only "
                "breadsticks can go on it, since the driver carries spare ones."
            )
        self.check_availability(item.id)
        line = next((ln for ln in self.lines if ln.item.id == item.id), None)
        if line is None and quantity == 0:
            raise PosError(f"There are no {item.name} on the order to take off.")
        if not 0 <= quantity <= MAX_EXTRAS:
            raise PosError(f"The driver carries at most {MAX_EXTRAS} orders of {item.name}.")
        before = self.quote.total if self.quote else Decimal("0")
        if line is None:
            line = Line(f"L{self._next_line}", item, item.sizes[0], quantity)
            self._next_line += 1
            self.lines.append(line)
        elif quantity == line.quantity:
            return line, Decimal("0")
        elif quantity == 0:
            self.lines.remove(line)
        else:
            line.was, line.rev = line.describe().split(": ", 1)[1], line.rev + 1
            line.quantity = quantity
        self.last_touched = line.id
        self._touch()
        diff = self.price().total - before
        if quantity:
            # what the extra adds to the original order, for the read-back
            assert self.quote is not None
            extra = self.quote.total - self._quote(self.subtotal() - line.total).total
            self._changed(
                "extra", f"Added {line.describe().split(': ', 1)[1]}, ${extra} more at the door"
            )
        else:
            self._changed("extra", None)
        return line, diff

    def set_driver_note(self, note: str) -> str:
        d = self._open()
        note = _driver_note(note)
        if looks_like_an_address(note):
            # the order is on its way to the address on it: a note can't send it elsewhere
            raise PosError(
                f"A driver note can't change where the order goes: {d.driver} is taking it to "
                f"{self.fulfillment.address}, the address on the order, and that can't change. "
                "Say so plainly, and never say the address was updated. If they need it "
                "somewhere else, they can cancel the order (cancel_order)."
            )
        d.note = note
        self._changed("note", f"Driver note: {note}")
        return note

    def read_back(self) -> Delivery:
        d = self._live()
        cancelled = self.order is not None and self.order.cancelled
        if not cancelled and (out := self.out_of_stock()):
            raise PosError(
                f"The {out[0].item.name} is still unresolved. Ask: swap it ("
                f"{', '.join(self.same_price_swaps(out[0]))}) or take it off the order."
            )
        d.read_back_rev = d.rev
        return d

    def end_without_update(self, reason: EndedEarly) -> Delivery:
        # wrong person, call back later, or not interested: the order goes on as it is
        d = self._en_route()
        d.ended_early = reason
        return d

    @property
    def read_back_current(self) -> bool:
        d = self.delivery
        return d is not None and d.read_back_rev == d.rev

    @property
    def call_done(self) -> bool:
        # the customer heard the changes read back, or the call ended without the update
        d = self.delivery
        return d is not None and (self.read_back_current or d.ended_early is not None)

    def cart_summary(self) -> str:
        if not self.lines:
            return "The cart is empty."
        return "Cart: " + "; ".join(ln.describe() for ln in self.lines) + "."

    def snapshot(self) -> dict[str, Any]:
        q = self.quote
        return {
            "menu": [
                {
                    "id": m.id,
                    "name": m.name,
                    "category": m.category,
                    "blurb": m.blurb,
                    "sizes": [
                        {"id": s.id, "label": s.label, "price": float(s.price)} for s in m.sizes
                    ],
                    "allergens": sorted(m.allergens),
                    # wing sauces that carry an allergen, e.g. the Thai sauce's peanuts
                    "sauceAllergens": (
                        {s.label: sorted(s.allergens) for s in SAUCES if s.allergens}
                        if m.needs_sauce
                        else {}
                    ),
                    "available": m.available,
                }
                for m in self.menu
            ],
            "cart": {
                "version": self.version,
                "lines": [ln.to_dict() for ln in self.lines],
                "removed": [ln.to_dict() for ln in self.removed],
                "subtotal": float(self.subtotal()),
                "lastTouched": self.last_touched,
            },
            "fulfillment": {
                "mode": self.fulfillment.mode,
                "address": self.fulfillment.address,
                "zip": self.fulfillment.zip,
                "note": self.fulfillment.note,
            },
            # the delivery update's customer, when no name is known, is just "Customer"
            "customer": {
                "name": "Customer" if self.delivery and not self.name else self.name,
                "phone": self.phone,
            },
            "allergies": [
                {"allergen": a, "severe": severe} for a, severe in sorted(self.allergies.items())
            ],
            "upsellOffered": self.upsell_offered,
            "quote": (
                {
                    "subtotal": float(q.subtotal),
                    "tax": float(q.tax),
                    "deliveryFee": float(q.delivery_fee),
                    "smallOrderFee": float(q.small_order_fee),
                    "tip": float(q.tip),
                    "total": float(q.total),
                    "current": self.quote_current,
                }
                if q
                else None
            ),
            "order": (
                {
                    "number": self.order.number,
                    "readyIn": self.order.ready_in,
                    "cancelled": self.order.cancelled,
                }
                if self.order
                else None
            ),
            "payment": self.payment(),
            "delivery": self._delivery_snapshot(),
        }

    def _delivery_snapshot(self) -> dict[str, Any] | None:
        d = self.delivery
        if d is None:
            return None
        return {
            "driver": d.driver,
            "etaMinutes": d.eta_minutes,
            "due": float(d.due),
            "dueNow": float(self.due_now()),
            "balance": float(self.balance()),
            "note": d.note,
            "cancelled": bool(self.order and self.order.cancelled),
            "outOfStock": [ln.id for ln in self.out_of_stock()],
            "changes": list(d.changes.values()),
            "readBack": self.read_back_current,
            "endedEarly": ENDED_EARLY[d.ended_early] if d.ended_early else None,
        }


# a street address: a number, then a street name ending in a street word ("500 Howard Street")
_STREET = re.compile(
    r"\b\d+[a-z]?\s+(?:[a-z0-9'.-]+\s+){0,3}(?:st|street|ave|avenue|blvd|boulevard|rd|road|way|"
    r"dr|drive|ln|lane|ct|court|pl|place|ter|terrace|hwy|highway|pkwy|parkway|sq|square)\b",
    re.IGNORECASE,
)


def looks_like_an_address(note: str) -> bool:
    """Whether a driver note names a street address, i.e. would send the order elsewhere."""
    return bool(_STREET.search(note))


def _driver_note(note: str) -> str:
    note = " ".join(note.split())
    if not note:
        raise PosError("The note is empty. Ask what the driver should know.")
    if len(note) > NOTE_MAX:
        raise PosError(
            f"That's too long for the driver's note (keep it under {NOTE_MAX} characters): ask "
            "them for the key details in a few words."
        )
    return note


def _check_quantity(quantity: int) -> None:
    if not 1 <= quantity <= MAX_QUANTITY:
        raise PosError(f"Quantity has to be between 1 and {MAX_QUANTITY}.")


def _check_pieces(size: Size, quantity: int) -> None:
    # "a dozen wings" comes through as twelve orders of a dozen more often than you'd think
    if size.pieces and quantity == size.pieces:
        raise PosError(
            f"Quantity is how many orders: a {size.label} is one order of {size.pieces}. "
            f"For a {size.label}, use quantity 1."
        )


def demo_pos(direction: str = "inbound", name: str = "") -> Pos:
    pos = Pos(menu=seed_menu())
    if direction == "outbound":
        # no name means none is known: the order stays under no name, never a made-up one
        _seed_delivery(pos, name)
    return pos


def _seed_delivery(pos: Pos, name: str) -> None:
    # order #214, placed earlier and due at the door (cash or card to the driver); the Sprite
    # ran out after it went in
    pos.add("pepperoni", 1, "large")
    pos.add("wings", 1, "dozen", sauce="hot")
    pos.add("sprite", 1, "large")
    pos.set_delivery("300 De Haro St, 94107")
    pos.set_contact(name, "4155550142")
    q = pos.price()
    pos.order = Order(number="214", ready_in=f"out with {DRIVER}, about {ETA_MINUTES} minutes away")
    pos.item("sprite").available = False
    pos.last_touched = None
    pos.delivery = Delivery(driver=DRIVER, eta_minutes=ETA_MINUTES, due=q.total)
