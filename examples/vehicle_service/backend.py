"""Bayside BMW's mock service desk, which enforces the booking rules, and (on the outbound
call) the repair order for a car that's in the shop today."""

from __future__ import annotations

import calendar
import dataclasses
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ServiceId = Literal["oil_change", "tire_rotation", "diagnostic", "warranty_repair"]
Transport = Literal["loaner", "shuttle", "wait"]
RecallDecision = Literal["added", "declined"]
# the consultation: what Nick asks about a problem, one technician note for each answer
NoteKind = Literal["sound", "where", "when", "since", "warning_lights", "recent", "other"]
# the follow-up call: what the customer says about each repair the technician found
RepairDecision = Literal["approved", "declined", "deferred"]
Urgency = Literal["now", "next_visit"]
EndedEarly = Literal["wrong_person", "call_back_later"]
Ride = Literal["shuttle", "self"]
# an appointment that's booked: the one on file from before the call, or the one booked on it
Booking = Literal["existing", "new"]
BookingStatus = Literal["booked", "cancelled"]

WARRANTY_YEARS = 3
WARRANTY_MILES = 36_000
LOANER_MIN_HOURS = 3.0
SHOP_CLOSES = dt.time(17, 30)
BOOKING_DAYS = 5
SLOT_TIMES = ((7, 30), (8, 0), (9, 0), (10, 30), (13, 0), (14, 30), (16, 0))
ADVISORS = ("Dana Whitfield", "Marcus Lee", "Priya Nair")
TRANSPORT_LABELS: dict[str, str] = {
    "loaner": "Loaner car",
    "shuttle": "Courtesy shuttle",
    "wait": "Wait in the lounge",
}
NOTE_LABELS: dict[str, str] = {
    "sound": "Sound",
    "where": "Where",
    "when": "When",
    "since": "Since",
    "warning_lights": "Warning lights",
    "recent": "Recent work",
    "other": "Note",
}
# the follow-up call: the X3 came in at 7:30 for the diagnostic and the recall
DROPPED_OFF = dt.time(7, 30)
LABOR_RATE = 118.0  # dollars an hour
SAME_DAY_HOURS = 2.0  # approved work this long or shorter is done by closing
NEXT_DAY_READY = dt.time(10, 0)  # longer work is ready the next business morning
SHUTTLE_LEAD = dt.timedelta(minutes=30)  # the shuttle leaves this long before the car is ready
SHUTTLE_MILES = 10
DECISIONS: dict[RepairDecision, str] = {
    "approved": "Approved",
    "declined": "Declined",
    "deferred": "Thinking it over",
}
URGENCY: dict[Urgency, str] = {"now": "Needs doing now", "next_visit": "Can wait"}
ENDED_EARLY: dict[EndedEarly, str] = {
    "wrong_person": "Someone else answered",
    "call_back_later": "Asked to call back later",
}
RIDES: dict[Ride, str] = {"shuttle": "Courtesy shuttle", "self": "Coming in themselves"}
# the dealership is in the Bay Area: "today" and "tomorrow" are its days
TIMEZONE = ZoneInfo("America/Los_Angeles")

_VEHICLE_STOPWORDS = {"bmw", "the", "my", "a", "an", "series", "car", "one"}


_STOP_ADVICE = (
    "Safety first, before anything else (no recall or other offers yet): tell them clearly "
    "and seriously not to drive the car, and that it should be towed in, not driven. You "
    "can't arrange a tow from the desk, so say that plainly. Offer the earliest open slot."
)
# where the car is picked up from on the follow-up, if not by the customer
PICKUP_BY_SOMEONE_ELSE = (
    "Anyone they send can collect the car: that person gives the customer's name or the repair "
    "order number at the service desk, and pays there."
)
# how a customer pays, the only answer to "can I pay by card?" or "by text?"
PAYMENT = (
    "Customers pay in person at the service desk when they pick the car up, by cash or card; "
    "recall work and warranty repairs are no charge. Nothing is paid over the phone or by text, and you never "
    "ask for or take card details: if they start reading a card number, stop them politely."
)


class ServiceError(Exception):
    """A rule the service desk won't break. The message is for the agent to relay."""


@dataclass(frozen=True)
class MenuItem:
    id: ServiceId
    label: str
    price: float
    hours: float
    # how the price reads: a flat menu price, the diagnostic fee only, or covered by warranty
    kind: Literal["menu", "fee", "warranty"]


MENU: dict[str, MenuItem] = {
    "oil_change": MenuItem("oil_change", "Oil change", 79.0, 1.0, "menu"),
    "tire_rotation": MenuItem("tire_rotation", "Tire rotation", 39.0, 0.5, "menu"),
    "diagnostic": MenuItem("diagnostic", "Diagnostic", 159.0, 2.0, "fee"),
    "warranty_repair": MenuItem("warranty_repair", "Warranty repair", 0.0, 4.0, "warranty"),
}

# what a caller might call each priced service, once filler, qualifiers, and plurals are
# dropped (see _quote_words). Anything not here has no price to give.
_QUOTE_ALIASES: dict[str, str] = {
    "oil": "oil_change",
    "oil change": "oil_change",
    "oil service": "oil_change",
    "oil and filter": "oil_change",
    "oil and filter change": "oil_change",
    "oil changed": "oil_change",
    "tire rotation": "tire_rotation",
    "rotation": "tire_rotation",
    "rotate tire": "tire_rotation",
    "rotate": "tire_rotation",
    "rotating tire": "tire_rotation",
    "tire rotated": "tire_rotation",
    "diagnostic": "diagnostic",
    "diagnosis": "diagnostic",
    "looked at": "diagnostic",
    "look at": "diagnostic",
    "inspection": "diagnostic",
    "warranty": "warranty_repair",
    "warranty repair": "warranty_repair",
    "warranty work": "warranty_repair",
}
# question words, the car, and qualifiers like "synthetic": none of them change the price
_QUOTE_FILLER = set(
    "how much is are does do will would what whats s it the a an any anything my our your for "
    "of on to be get cost costs price pricing charge fee quote rough roughly ballpark free "
    "extra pay paid included bmw car suv sedan series synthetic full regular standard basic "
    "conventional normal quick just x xdrive have having i me".split()
)
_TOTAL_WORDS = set(
    "total estimate altogether bill all in everything whole visit so far owe".split()
)
# how a caller asks for several menu items in one breath: "an oil change and a tire rotation"
_QUOTE_JOINERS = {"and", "plus", "with"}
# "whatever's earliest", "any day": no day to narrow to, so every open slot
_ANY_DAY = re.compile(
    r"\b(earliest|soonest|first available|next available|asap|any ?(day|time)|whenever|whatever)\b"
)
# "brakes", "braking", "pads", "tires": a wear item, by the word it starts with
_WEAR_ITEM = re.compile(r"\b(brak|pad|rotor|tire|tyre|wiper)\w*", re.IGNORECASE)
_WEAR_NOUN = {"brak": "brake"}
_DAY_WORDS = {
    name[:n].lower(): i for i, name in enumerate(calendar.day_name) for n in range(3, len(name) + 1)
} | {"weds": 2}
_MONTH_WORDS = {name[:3].lower(): i for i, name in enumerate(calendar.month_name) if name}
# "the 8th", "the 8" (but not "the 8 o'clock"), "1st"
_DAY_OF_MONTH = re.compile(
    r"\bthe\s+(\d{1,2})\b(?!\s*(?:o.?clock|:|[ap]\.?m))|\b(\d{1,2})(?:st|nd|rd|th)\b"
)


@dataclass
class Recall:
    id: str
    title: str
    hours: float
    parts_in_stock: bool = True


@dataclass(frozen=True)
class RecallRule:
    """An open recall for every car of a model family built in a range of model years."""

    family: str  # "X3", as bmw_model() names it
    first_year: int
    last_year: int
    recall: Recall


# the open recalls the desk knows about; a car on no rule here has none
RECALLS = (
    RecallRule("X3", 2023, 2023, Recall("23V-418", "Rear-view camera software update", 0.5)),
)


def recalls_for(family: str, year: int) -> list[Recall]:
    """The open recalls on a car, from the catalog: each its own copy."""
    return [
        dataclasses.replace(r.recall)
        for r in RECALLS
        if r.family == family and r.first_year <= year <= r.last_year
    ]


@dataclass
class Vehicle:
    vin: str
    year: int
    name: str  # "X3 xDrive30i", as BMW badges it
    body: Literal["suv", "sedan"]
    color: str
    miles: int
    in_service: dt.date
    recalls: list[Recall] = field(default_factory=list)
    miles_on_file: int = 0
    added: bool = False  # put on the file on this call (add_vehicle)

    @property
    def label(self) -> str:
        return f"{self.year} BMW {self.name}"

    def age_years(self, today: dt.date) -> float:
        return (today - self.in_service).days / 365.25

    def in_warranty(self, today: dt.date) -> bool:
        return self.age_years(today) < WARRANTY_YEARS and self.miles < WARRANTY_MILES

    def warranty_reason(self, today: dt.date) -> str:
        age = self.age_years(today)
        years = spoken_span(round(age * 2) / 2, "year")
        if age >= WARRANTY_YEARS:
            return f"Out of warranty: about {years} in service, past the {WARRANTY_YEARS}-year mark"
        if self.miles >= WARRANTY_MILES:
            return f"Out of warranty: {self.miles:,} miles, past the {WARRANTY_MILES:,}-mile limit"
        return f"In warranty: about {years} in service and {self.miles:,} miles"


@dataclass
class Customer:
    id: str
    name: str
    phone: str  # ten digits
    vehicles: list[Vehicle]
    # set up on this call, for a name and number with no file: no cars, nothing booked
    new: bool = False
    # already booked on the file before the call
    on_file: Appointment | None = None


@dataclass
class Slot:
    id: str  # "wed-0730": the weekday and time, which is how callers ask for them
    start: dt.datetime
    advisor: str
    open: bool = True

    def fits(self, hours: float) -> bool:
        end = self.start + dt.timedelta(hours=hours)
        return end.time() <= SHOP_CLOSES and end.date() == self.start.date()

    def spoken(self) -> str:
        return f"{self.start:%A %B} {self.start.day}, {clock(self.start)}"


@dataclass
class QuoteAsk:
    asked: str
    price: float | None
    answer: str


@dataclass
class TechNote:
    """One thing the caller said about the problem, for the technician at drop-off."""

    kind: NoteKind
    text: str

    @property
    def label(self) -> str:
        return NOTE_LABELS[self.kind]


@dataclass(frozen=True)
class LikelyCause:
    """What a sound at a moment often points to: said cautiously, never as a diagnosis."""

    sounds: tuple[str, ...]  # word stems in the sound note
    moments: tuple[str, ...]  # word stems in the when note
    cause: str
    unsafe: bool
    advice: str


LIKELY_CAUSES = (
    LikelyCause(
        ("grind",),
        ("brak", "stop"),
        "brake pads worn down to the metal",
        True,
        "Grinding brakes may not stop the car well: tell them plainly to drive it as little as "
        "possible, and offer the soonest drop-off.",
    ),
    LikelyCause(
        ("squeal", "squeak", "screech"),
        ("brak", "stop"),
        "brake pads near the end of their life (the wear indicator squeals)",
        False,
        "Fine to drive in the meantime, but worth looking at soon.",
    ),
    LikelyCause(
        ("clunk", "knock", "thump"),
        ("bump", "pothole", "speed bump", "turn"),
        "a worn suspension part, like a sway bar link or a control arm bushing",
        False,
        "Usually fine to drive carefully until it's looked at.",
    ),
    LikelyCause(
        ("hum", "whine", "roar", "drone", "growl"),
        ("speed", "highway", "freeway", "mph", "faster"),
        "a worn wheel bearing, or uneven tire wear",
        False,
        "Fine to drive gently, but get it looked at soon: a bearing gets louder as it wears.",
    ),
    LikelyCause(
        ("click",),
        ("turn",),
        "a worn CV joint on the axle",
        False,
        "Fine to drive gently until it's looked at.",
    ),
    LikelyCause(
        ("rattl",),
        ("idle", "cold", "start"),
        "a loose heat shield under the car",
        False,
        "Usually harmless, and fine to drive.",
    ),
)
# warning lights that mean stop driving, whatever the noise
_STOP_LIGHTS = re.compile(r"\b(brake|oil|temperature|temp|coolant|overheat\w*)\b", re.IGNORECASE)
_NO_LIGHTS = re.compile(r"^\s*(no|none|nothing|nope)\b", re.IGNORECASE)
# what the caller says that means the car shouldn't be driven at all, in a note or the concern
_CANT = r"(?:won'?t|doesn'?t|can'?t|didn'?t|isn'?t|not|barely)"
_STOP_DRIVING = (
    (
        re.compile(
            rf"\bbrakes?\b[^.]*\b(fail\w*|gone|went out|not working|{_CANT} work)\b|"
            rf"\bno brakes\b|\bpedal\b[^.]*\b(floor|sink\w*)\b|"
            rf"\b{_CANT}\s+stop(ping)?\s+(safely|properly|right|in time|well)\b|"
            rf"\b(car|it)\s+{_CANT}\s+stop\b",
            re.IGNORECASE,
        ),
        "the brakes aren't stopping it",
    ),
    (
        re.compile(r"(?<!\bno )(?<!\bnot )\b(smok\w*|burning|fire|flames?)\b", re.IGNORECASE),
        "smoke or burning",
    ),
    (
        re.compile(
            r"\b(lost|losing|lose|no|can'?t|couldn'?t)\s+(the\s+)?(power\s+)?steer\w*|"
            r"\bsteer\w*\b[^.]*\b(went out|gave out|locked|locking|fail\w*|gone)\b",
            re.IGNORECASE,
        ),
        "losing steering",
    ),
    (
        re.compile(
            r"\b(flash\w*|blink\w*|red)\b[^.]*\b(light|warning)\b|"
            r"\b(light|warning)\b[^.]*\b(flash\w*|blink\w*|red)\b",
            re.IGNORECASE,
        ),
        "a flashing or red warning light",
    ),
    (
        re.compile(
            r"\b(overheat\w*|steam\w*|temp(erature)?\s+(gauge|light|warning)|coolant light)\b",
            re.IGNORECASE,
        ),
        "overheating",
    ),
)
# what a warranty repair is never for: wear, a noise from the brakes, maintenance, the recall
_NOT_A_DEFECT = re.compile(
    r"\b(grind\w*|squeal\w*|squeak\w*|recall\w*|oil (change|service)|maintenance|rotation|"
    r"tune.?up)\b",
    re.IGNORECASE,
)


@dataclass
class Draft:
    vin: str | None = None
    services: list[str] = field(default_factory=list)
    concern: str = ""
    notes: list[TechNote] = field(default_factory=list)
    recall: RecallDecision | None = None
    slot_id: str | None = None
    transport: Transport | None = None
    approval_limit: float | None = None  # "call me before anything over $500"
    revision: int = 0
    read_back_revision: int | None = None


@dataclass
class Appointment:
    confirmation: str
    vin: str
    vehicle: str
    services: list[str]
    concern: str
    recall_id: str | None
    recall: RecallDecision | None
    slot_id: str
    when: str
    advisor: str
    transport: Transport
    hours: float
    estimate: float
    notes: list[TechNote] = field(default_factory=list)
    status: BookingStatus = "booked"
    moved_from: str | None = None  # when it was before it was last moved
    approval_limit: float | None = None
    stop_driving: str | None = None  # why it shouldn't be driven in, if it shouldn't

    @property
    def what(self) -> str:
        """ "tire rotation on the 330i", the way it's said."""
        # the label is "2019 BMW 330i": the model is the third word ("3 Series": two words)
        model = self.vehicle.split()[2:]
        car = " ".join(model[:2]) if model[0].isdigit() and len(model) > 1 else model[0]
        return f"{' and '.join(self.services).lower()} on the {car}"


@dataclass(frozen=True)
class Pending:
    """A move or a cancellation that's been read back and waits for the caller's yes."""

    action: Literal["move", "cancel"]
    booking: Booking
    slot_id: str | None = None


@dataclass(frozen=True)
class Finding:
    """What the technician found, priced like the menu: parts plus labor at the shop rate."""

    id: str
    title: str  # "Front brake pads and rotors"
    found: str  # what the technician saw
    why: str  # why it needs doing, or why it can wait
    parts: float
    parts_detail: str  # the parts, priced, e.g. "pads $138, rotors $174"
    labor_hours: float
    urgency: Urgency
    soon: str  # how soon, in the technician's words
    covered: str  # why warranty doesn't pay for it
    cheaper: str  # a cheaper option the shop allows, or why there isn't one

    @property
    def labor(self) -> float:
        return self.labor_hours * LABOR_RATE

    @property
    def price(self) -> float:
        return self.parts + self.labor


# what the technician found on the X3 the caller brought in for a grinding noise when braking
FINDINGS = (
    Finding(
        id="front_brakes",
        title="Front brake pads and rotors",
        found=(
            "The front pads are worn through to the metal backing plate, and that's scored "
            "both front rotors. That's the grinding."
        ),
        why=(
            "Metal on metal stops the car worse every mile, and scored rotors can't be "
            "resurfaced, so the pads and rotors are replaced together."
        ),
        parts=312.0,
        parts_detail="front pads $138, two front rotors $174, genuine BMW parts",
        labor_hours=1.5,
        urgency="now",
        soon="It shouldn't be driven like this: it needs doing before the car's back on the road.",
        covered="Not covered by warranty: brakes are a wear item.",
        cheaper=(
            "There's no cheaper option: with the rotors scored, new pads alone wouldn't stop "
            "right, and there's no discount on this repair."
        ),
    ),
    Finding(
        id="rear_brakes",
        title="Rear brake pads",
        found="The rear pads are at 4 millimeters; new is 10.",
        why="They're wearing normally and still have some life left.",
        parts=96.0,
        parts_detail="rear pads $96, genuine BMW parts",
        labor_hours=1.0,
        urgency="next_visit",
        soon=("Not an emergency: fine to do at the next visit, in the next 3,000 to 5,000 miles."),
        covered="Not covered by warranty: brakes are a wear item.",
        cheaper="No cheaper option, but there's no need to do them today.",
    ),
)


@dataclass
class ShopVisit:
    """The follow-up call: the visit booked for the noise is in the shop today, and what the
    technician found."""

    appointment: Appointment
    findings: list[Finding]
    # who picked up: unknown until they say, then the customer or someone else
    reached: Literal["unknown", "customer", "someone_else"] = "unknown"
    ended_early: EndedEarly | None = None
    decisions: dict[str, RepairDecision] = field(default_factory=dict)  # finding id -> decision
    next_visit: list[str] = field(default_factory=list)  # notes for the next visit
    questions: list[str] = field(default_factory=list)  # asked about the repairs on this call
    ride: Ride | None = None
    pickup_from: str = ""  # where the shuttle picks them up
    collected_by: str = ""  # someone else picking the car up for them
    concerns: list[str] = field(default_factory=list)  # something new they've noticed
    rev: int = 0  # bumped on every change, so a stale summary can be told from a current one
    summary_rev: int | None = None


class ServiceDesk:
    """The dealer's system for one call: the customer, the calendar, and the draft appointment."""

    def __init__(
        self, customers: list[Customer], slots: list[Slot], today: dt.date, ro_number: int
    ) -> None:
        self.today = today
        self._customers = customers
        self.slots = slots
        self.customer: Customer | None = None
        self.recalls_checked: set[str] = set()  # VINs checked on this call
        self.draft = Draft()
        self.quotes: list[QuoteAsk] = []
        self.appointment: Appointment | None = None  # booked on this call
        self.changing = False  # the booking made on this call is open for changes again
        self.pending: Pending | None = None
        self.visit: ShopVisit | None = None  # set only for a car that's already in the shop
        self._ro_number = ro_number
        self._bookings = 0  # on this call, so a booking after a cancellation gets a new number

    # ---- lookups ----

    @property
    def on_file(self) -> Appointment | None:
        """What's already booked on the caller's file from before the call, once they're
        looked up: never another customer's."""
        return self.customer.on_file if self.customer else None

    def lookup_customer(self, phone: str, name: str = "") -> Customer:
        """The file for this name and number. A name and number with no file get a new one,
        with no cars and nothing booked: a different name on a known number shares nothing
        from that number's file."""
        digits = re.sub(r"\D", "", phone)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        if len(digits) != 10:
            raise ServiceError(
                "That's only part of a phone number (US numbers are ten digits). Ask them for "
                "the whole number again, with the area code. Never fill in or guess digits."
            )
        if not real_us_number(digits):
            raise ServiceError(
                "That isn't a real phone number. Ask them for it again, with the area code; "
                "never fill in or guess digits."
            )
        name = " ".join(name.split())
        if not name:
            raise ServiceError(
                "Ask for their name first (on its own), then call lookup_customer with the "
                "name and the number."
            )
        c = next(
            (c for c in self._customers if c.phone == digits and same_person(name, c.name)),
            None,
        )
        if c is None:
            c = Customer(
                id=f"C-{30_001 + len(self._customers)}",
                name=name,
                phone=digits,
                vehicles=[],
                new=True,
            )
            self._customers.append(c)
        if c is not self.customer:
            # the car and anything read back were the other file's
            self.draft.vin = None
            self.draft.recall = None
            self.pending = None
            self._changed()
        self.customer = c
        return c

    def vehicle(self, query: str) -> Vehicle:
        """A car on the customer's file, by VIN (or its last six) or by year and model."""
        c = self._require_customer()
        if not c.vehicles:
            raise ServiceError(
                "There are no cars on their file yet. Ask what car it is, one question at a "
                "time (the year and model, then the mileage), then call add_vehicle."
            )
        q = re.sub(r"[^a-z0-9 ]", " ", query.lower())
        compact = q.replace(" ", "")
        for v in c.vehicles:
            if len(compact) >= 6 and v.vin.lower().endswith(compact):
                return v
        words = [w for w in q.split() if w not in _VEHICLE_STOPWORDS]
        # "my 3" or "the three" could be the X3 or a 3 series: ask, unless something else says
        rest = [w for w in words if w not in {"3", "three"}]
        if (
            len(c.vehicles) > 1
            and len(rest) < len(words)
            and not any(_score(v, rest) for v in c.vehicles)
        ):
            names = " or the ".join(v.name.split()[0] for v in c.vehicles)
            raise ServiceError(f'"3" could be the {names}. Ask which one.')
        # the car the most words point at, so "actually it's the 330i" still lands
        scores = {v.vin: _score(v, words) for v in c.vehicles}
        best = max(scores.values())
        matches = [v for v in c.vehicles if scores[v.vin] == best]
        if best > 0 and len(matches) == 1:
            return matches[0]
        on_file = ", ".join(v.label for v in c.vehicles)
        raise ServiceError(
            f"Couldn't tell which car they mean. On file: {on_file}. If it's a car that isn't "
            "on their file, ask the year and model, then the mileage, and call add_vehicle."
        )

    def add_vehicle(self, year: int, model: str, miles: int, vin: str = "") -> Vehicle:
        """A car the caller describes, put on their file and selected for the visit. Its
        warranty comes from the model year and the miles, and its recalls from the catalog."""
        self._require_open()
        c = self._require_customer()
        found = bmw_model(model)
        if found is None:
            raise ServiceError(
                f"{model!r} isn't a BMW model this desk services: this desk only services "
                "BMWs. Say so plainly and kindly. If it is a BMW, ask which model (an X5, a "
                "330i, an i4)."
            )
        name, family = found
        if not 1980 <= year <= self.today.year + 1:
            raise ServiceError("That model year doesn't look right. Ask them for the year again.")
        if not 0 <= miles < 500_000:
            raise ServiceError("That mileage doesn't look right. Ask them to read the odometer.")
        same = next((v for v in c.vehicles if v.year == year and _family(v) == family), None)
        if same is not None:
            raise ServiceError(
                f"The {same.label} is already on their file: call select_vehicle with it (and "
                "update_mileage if the mileage has changed) instead."
            )
        typed = re.sub(r"[^A-Za-z0-9]", "", vin).upper()
        taken = {v.vin for x in self._customers for v in x.vehicles}
        if len(typed) != 17 or typed in taken:
            # no VIN is needed: one the desk makes up stands in for it
            n = len(taken) + 1
            typed = f"NEW{n:014d}"
            while typed in taken:
                n += 1
                typed = f"NEW{n:014d}"
        v = Vehicle(
            vin=typed,
            year=year,
            name=name,
            body="suv" if family.startswith(("X", "iX")) else "sedan",
            color="",
            miles=miles,
            # the date it went into service isn't known: take July 1 of the model year (never
            # later than today), so the warranty's 3 years run from the middle of that year
            in_service=min(dt.date(year, 7, 1), self.today),
            recalls=recalls_for(family, year),
            added=True,
        )
        c.vehicles.append(v)
        self.draft.vin = v.vin
        self.draft.recall = None
        self._changed()
        self._revalidate()
        return v

    def check_recalls(self, vin: str) -> list[Recall]:
        v = self._vehicle_by_vin(vin)
        self.recalls_checked.add(v.vin)
        return list(v.recalls)

    def quote(self, service: str) -> QuoteAsk:
        """A price, but only for what's on the menu; never a made-up estimate."""
        words = _quote_words(re.sub(r"[,&+]", " and ", service))
        item = MENU.get(_QUOTE_ALIASES.get(" ".join(words), ""))
        several = _menu_items(words) if item is None else []
        if len(several) == 1:
            item = several[0]
        free = self._free_answer(set(words))
        if len(several) > 1:
            total = sum(m.price for m in several)
            lines = ", ".join(f"{m.label.lower()} ${m.price:.0f}" for m in several)
            after = (
                " Any repair is quoted after the inspection."
                if any(m.kind == "fee" for m in several)
                else ""
            )
            ask = QuoteAsk(asked=service, price=total, answer=f"${total:.0f}: {lines}.{after}")
        elif item is None and free is not None:
            ask = QuoteAsk(asked=service, price=0.0, answer=free)
        elif item is None and set(words) <= _TOTAL_WORDS:
            ask = QuoteAsk(
                asked=service,
                price=self.estimate() if self.draft.services else None,
                answer=self._estimate_answer(),
            )
        elif item is None:
            ask = QuoteAsk(
                asked=service,
                price=None,
                answer=(
                    f"No set price for {service!r}. Only the oil change ($79) and tire rotation "
                    "($39) have menu prices, and diagnostics are a $159 fee. Don't estimate. "
                    "Offer to book a diagnostic: the technician quotes any repair after it."
                ),
            )
        elif item.kind == "warranty":
            ask = QuoteAsk(
                asked=service,
                price=0.0,
                answer="Covered at no charge when the car is in warranty: under three years "
                "and 36,000 miles.",
            )
        elif item.kind == "fee":
            ask = QuoteAsk(
                asked=service,
                price=item.price,
                answer="$159 diagnostic fee. The repair itself is quoted after the inspection.",
            )
        else:
            ask = QuoteAsk(asked=service, price=item.price, answer=f"${item.price:.0f} flat.")
        self.quotes.append(ask)
        return ask

    def _free_answer(self, words: set[str]) -> str | None:
        if "recall" in words:
            return "No charge: recall work is always free, parts and labor."
        if "shuttle" in words:
            return "The courtesy shuttle is free for anyone."
        if words & {"lounge", "wait", "waiting"}:
            return "Waiting in the lounge is free."
        if words & {"loaner", "rental"}:
            rule = (
                f"A loaner is free, but only for warranty or recall visits over "
                f"{LOANER_MIN_HOURS:g} hours."
            )
            if not self.draft.services:
                return rule
            if self.loaner_eligible():
                return f"{rule} This visit qualifies."
            return f"{rule} This visit doesn't. The shuttle is free."
        return None

    def _estimate_answer(self) -> str:
        if not self.draft.services:
            return "Nothing's on the visit yet, so there's no total. Find out what it needs first."
        priced = [MENU[s] for s in self.draft.services]
        lines = ", ".join(
            f"{m.label.lower()} " + (f"${m.price:.0f}" if m.price else "no charge") for m in priced
        )
        total = self.estimate()
        answer = f"Estimate ${total:.0f}: {lines}." if total else f"No charge: {lines}."
        if self.draft.recall == "added" and self.open_recall():
            answer += " The recall is no charge."
        if "diagnostic" in self.draft.services:
            answer += " Any repair is quoted after the inspection."
        return answer

    # ---- the draft: one field at a time, so a correction is cheap ----

    def select_vehicle(self, query: str) -> list[str]:
        self._require_open()
        v = self.vehicle(query)
        if self.draft.vin != v.vin:
            self.draft.vin = v.vin
            # the recall decision was about the other car
            self.draft.recall = None
            self._changed()
        return self._revalidate()

    def set_mileage(self, miles: int) -> list[str]:
        self._require_open()
        v = self.require_vehicle()
        if not 0 < miles < 500_000:
            raise ServiceError("That mileage doesn't look right. Ask them to read the odometer.")
        if miles < v.miles_on_file:
            raise ServiceError(
                f"The file already shows {v.miles_on_file:,} miles, and mileage only goes up. "
                "Ask them to check the odometer."
            )
        if v.miles != miles:
            v.miles = miles
            self._changed()
        return self._revalidate()

    def set_services(
        self, services: list[str], concern: str = "", warranty_agreed: bool = False
    ) -> list[str]:
        self._require_open()
        v = self.require_vehicle()
        if not services:
            raise ServiceError("Pick at least one service.")
        unknown = [s for s in services if s not in MENU]
        if unknown:
            raise ServiceError(f"Not a service we book: {', '.join(unknown)}.")
        services = list(dict.fromkeys(services))
        # the concern set_concern summed up, unless this call names its own
        concern = " ".join(concern.split()) or self.draft.concern
        warranty = "warranty_repair" in services
        # the concern and every note: a grinding note doesn't become warranty work by being
        # left out of the summary
        said = " ; ".join([concern, *(n.text for n in self.draft.notes)])
        wear = _WEAR_ITEM.search(said) if warranty else None
        if wear:
            noun = _WEAR_NOUN.get(wear[1].lower(), wear[1].lower())
            rotation = ", or a tire rotation ($39)" if noun in {"tire", "tyre"} else ""
            raise ServiceError(
                f"{noun.capitalize()}s are wear items, never covered by warranty. Offer it "
                f"as a regular visit at its price: a diagnostic ($159) to look at it{rotation}. "
                "Don't try warranty_repair again for this."
            )
        other = _NOT_A_DEFECT.search(said) if warranty else None
        if other:
            raise ServiceError(
                f"A warranty repair is only for a part that's failed (a liftgate, a window, "
                f"electronics, a leak), never for {other[0].lower()!r}: the recall goes on with "
                "decide_recall, maintenance is a menu service, and a noise is a diagnostic "
                "($159). Don't try warranty_repair again for this."
            )
        if warranty and not warranty_agreed and "warranty_repair" not in self.draft.services:
            raise ServiceError(
                "Only add a warranty repair once they've asked for warranty work, or you've "
                "explained it would be covered at no charge and they've agreed. Then call "
                "set_services again with warranty_agreed true."
            )
        if "warranty_repair" in services and not v.in_warranty(self.today):
            raise ServiceError(
                f"The {v.name} isn't in warranty. {v.warranty_reason(self.today)}. Offer a "
                "diagnostic instead, $159."
            )
        needs_concern = {"diagnostic", "warranty_repair"} & set(services)
        if needs_concern and not concern:
            raise ServiceError(
                "Call set_concern first with the problem summed up in a few words, for the "
                "technician. Use what they've already told you; ask only if they haven't said."
            )
        if services != self.draft.services or concern != self.draft.concern:
            self.draft.services = services
            self.draft.concern = concern
            self._changed()
        return self._revalidate() + ([self._duplicate_reason()] if self.duplicates() else [])

    def set_approval_limit(self, amount: float) -> None:
        """ "Don't do anything over $500 without calling me": on the booking and its read-back.
        It promises nothing about the price."""
        self._require_open()
        if not amount > 0:
            raise ServiceError("Pass the dollar amount they gave, e.g. 500.")
        if amount != self.draft.approval_limit:
            self.draft.approval_limit = float(amount)
            self._changed()

    def add_note(self, kind: str, text: str) -> TechNote:
        """A technician note: one per kind (a correction replaces it), any number of "other"."""
        self._require_unbooked()
        if kind not in NOTE_LABELS:
            raise ServiceError(f"Note kind is one of: {', '.join(NOTE_LABELS)}.")
        text = " ".join(text.split())
        if not text:
            raise ServiceError("Pass what they said, in a few words, as the note's text.")
        note = TechNote(cast(NoteKind, kind), text)
        if kind != "other":
            self.draft.notes = [n for n in self.draft.notes if n.kind != kind]
        self.draft.notes.append(note)
        return note

    def set_concern(self, summary: str) -> None:
        """The problem summed up for the technician; it's read back with the visit."""
        self._require_unbooked()
        summary = " ".join(summary.split())
        if not summary:
            raise ServiceError(
                'Pass the problem summed up in a few words, e.g. "grinding when braking, front".'
            )
        if summary != self.draft.concern:
            self.draft.concern = summary
            self._changed()

    def stop_driving(self) -> str | None:
        """Why the car shouldn't be driven at all, from what the caller has said, if it
        shouldn't: failing brakes, smoke, losing steering, a red or flashing light,
        overheating, or a brake, oil, or temperature light."""
        for text in [*(n.text for n in self.draft.notes), self.draft.concern]:
            if not text or _NO_LIGHTS.search(text):
                continue
            for pattern, why in _STOP_DRIVING:
                if pattern.search(text):
                    return why
        lights = next((n.text for n in self.draft.notes if n.kind == "warning_lights"), "")
        if lights and not _NO_LIGHTS.search(lights) and _STOP_LIGHTS.search(lights):
            return "a brake, oil, or temperature warning light"
        return None

    def concern_left_out(self) -> str | None:
        """A warning light they mentioned that the concern summary doesn't."""
        lights = next((n.text for n in self.draft.notes if n.kind == "warning_lights"), "")
        concern = self.draft.concern.lower()
        if lights and not _NO_LIGHTS.search(lights) and "light" not in concern:
            return lights
        return None

    def likely_cause(self) -> LikelyCause | None:
        """What the notes so far often point to, if they point anywhere."""
        notes = {n.kind: n.text.lower() for n in self.draft.notes}
        sound, moment = notes.get("sound", ""), notes.get("when", "")
        why = self.stop_driving()
        stop = why is not None
        for c in LIKELY_CAUSES:
            if any(w in sound for w in c.sounds) and any(w in moment for w in c.moments):
                if stop and not c.unsafe:
                    return LikelyCause(c.sounds, c.moments, c.cause, True, _STOP_ADVICE)
                return c
        if stop:
            # said cautiously ("…, to confirm"), named after what they described
            causes = {
                "the brakes aren't stopping it": "a brake system fault",
                "smoke or burning": "something overheating or leaking",
                "losing steering": "a steering fault",
            }
            cause = causes.get(why or "", "")
            if not cause:
                cause = (
                    "the fault the warning light is flagging"
                    if why and "light" in why
                    else "a fault the technician needs to find"
                )
            return LikelyCause((), (), cause, True, _STOP_ADVICE)
        return None

    def decide_recall(self, add: bool) -> None:
        self._require_open()
        v = self.require_vehicle()
        if v.vin not in self.recalls_checked:
            raise ServiceError("Check recalls for this car first.")
        if not v.recalls:
            raise ServiceError(f"The {v.name} has no open recalls.")
        decision: RecallDecision = "added" if add else "declined"
        if self.draft.recall != decision:
            self.draft.recall = decision
            self._changed()
        self._revalidate()

    def open_slots(self, day: str | None = None, hours: float | None = None) -> list[Slot]:
        """Open slots that fit a visit of `hours` (this call's visit, by default)."""
        hours = self.visit_hours() if hours is None else hours
        slots = [s for s in self.slots if s.open and s.fits(hours)]
        if day:
            date = self.resolve_day(day)
            if date is None and _ANY_DAY.search(day.lower()):
                return slots
            if date is None:
                raise ServiceError(
                    f"Didn't catch the day in {day!r}. Try again with a weekday or a date like "
                    f'"Oct 8", or leave the day out for every open day. Open days are '
                    f"{self.open_days()}."
                )
            if not any(s.start.date() == date for s in self.slots):
                if date <= self.today:
                    why = "There are no same-day openings."
                elif date.weekday() >= 5:
                    why = f"The service drive is closed on weekends, so nothing {date:%A}."
                else:
                    why = (
                        f"{date:%A %B} {date.day} is further out than the calendar goes "
                        f"({BOOKING_DAYS} business days)."
                    )
                raise ServiceError(f"{why} Open days are {self.open_days()}.")
            slots = [s for s in slots if s.start.date() == date]
        return slots

    def open_days(self) -> str:
        hours = self.visit_hours()
        days = dict.fromkeys(s.start.date() for s in self.slots if s.open and s.fits(hours))
        return ", ".join(f"{d:%A %B} {d.day}" for d in days) or "none this week"

    def resolve_day(self, day: str) -> dt.date | None:
        """The date a caller means: "tomorrow", "next tuesday", "Oct 8", "10/8", "2026-10-08"."""
        text = day.strip().lower()
        if "day after tomorrow" in text:
            return self.today + dt.timedelta(days=2)
        if "tomorrow" in text:
            return self.today + dt.timedelta(days=1)
        if "today" in text:
            return self.today
        try:
            if m := re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
                return dt.date(int(m[1]), int(m[2]), int(m[3]))
            if m := re.search(r"\b(\d{1,2})/(\d{1,2})\b", text):
                return self._upcoming(int(m[1]), int(m[2]))
            m = re.search(r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", text)
            if m and m[1] in _MONTH_WORDS:
                return self._upcoming(_MONTH_WORDS[m[1]], int(m[2]))
            # "the 8th", "Thursday the 1st": the next date with that day of the month
            if m := _DAY_OF_MONTH.search(text):
                n = int(m[1] or m[2])
                y, mo = self.today.year, self.today.month
                if n < self.today.day:
                    y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
                return dt.date(y, mo, n)
        except ValueError:
            return None
        for word in re.findall(r"[a-z]+", text):
            if word in _DAY_WORDS:
                ahead = (_DAY_WORDS[word] - self.today.weekday()) % 7
                # today only if it has an opening; "next wednesday" is never today
                if ahead == 0 and ("next" in text or not self._open_on(self.today)):
                    ahead = 7
                return self.today + dt.timedelta(days=ahead)
        return None

    def _open_on(self, day: dt.date) -> bool:
        return any(s.open and s.start.date() == day for s in self.slots)

    def day_word(self, day: dt.date) -> str:
        """The day the way it's said relative to today: "today", "tomorrow", "this Friday",
        "next Tuesday"."""
        if day == self.today:
            return "today"
        if day == self.today + dt.timedelta(days=1):
            return "tomorrow"
        same_week = day.isocalendar()[:2] == self.today.isocalendar()[:2]
        return f"{'this' if same_week else 'next'} {day:%A}"

    def _upcoming(self, month: int, day: int) -> dt.date:
        # a date without a year is the next one on or after today
        d = dt.date(self.today.year, month, day)
        return d if d >= self.today else dt.date(self.today.year + 1, month, day)

    def choose_slot(self, slot_id: str) -> None:
        self._require_open()
        slot = self._slot(slot_id)
        if not self._free(slot):
            raise ServiceError(f"{slot.spoken()} is taken. Offer another time.")
        hours = self.visit_hours()
        if not slot.fits(hours):
            raise ServiceError(
                f"A visit of {spoken_span(hours)} won't finish by closing if it starts at "
                f"{clock(slot.start)}. Offer an earlier drop-off."
            )
        if self.draft.slot_id != slot.id:
            self.draft.slot_id = slot.id
            self._changed()

    def choose_transport(self, option: str) -> None:
        self._require_open()
        if option not in TRANSPORT_LABELS:
            raise ServiceError("The options are a loaner, the shuttle, or waiting.")
        # a loaner depends on how long the visit is, which depends on the services
        if option == "loaner" and not self.draft.services:
            raise ServiceError(
                "Set the services first: whether a loaner fits depends on how long it takes."
            )
        if option == "loaner" and not self.loaner_eligible():
            raise ServiceError(self._no_loaner_reason())
        if self.draft.transport != option:
            self.draft.transport = cast(Transport, option)
            self._changed()

    def read_back(self) -> str:
        """The whole appointment, as the agent must say it. Marks this revision read back."""
        self._require_open()
        missing = self.missing()
        if missing:
            raise ServiceError(f"Not ready to read back. Still need: {', '.join(missing)}.")
        if left := self.concern_left_out():
            raise ServiceError(
                f"The concern leaves out the warning light they mentioned ({left}). Call "
                "set_concern again with every problem they've raised, e.g. "
                f'"{self.draft.concern}; {left}", then read_back.'
            )
        self.draft.read_back_revision = self.draft.revision
        return self.summary()

    def book(self) -> Appointment:
        self._require_open()
        v = self.require_vehicle()
        if v.vin not in self.recalls_checked:
            raise ServiceError("Check recalls for this car before booking.")
        missing = self.missing()
        if missing:
            raise ServiceError(f"Can't book yet. Still need: {', '.join(missing)}.")
        if self.draft.transport == "loaner" and not self.loaner_eligible():
            raise ServiceError(self._no_loaner_reason())
        if "warranty_repair" in self.draft.services and not v.in_warranty(self.today):
            raise ServiceError(f"The {v.name} isn't in warranty any more. Book a diagnostic.")
        if self.duplicates():
            raise ServiceError(self._duplicate_reason())
        slot = self._slot(self.draft.slot_id or "")
        if not self._free(slot) or not slot.fits(self.visit_hours()):
            raise ServiceError("That time isn't available for this visit any more.")
        if self.draft.read_back_revision != self.draft.revision:
            raise ServiceError(
                "The appointment changed since it was read back. Call read_back and read it "
                "to them again before booking."
            )
        assert self.draft.transport is not None
        recall = v.recalls[0] if v.recalls else None
        # a change to the booking made on this call keeps its number, and frees its old time
        before = self.appointment if self.changing else None
        if before:
            self._slot(before.slot_id).open = True
        else:
            self._bookings += 1
        slot.open = False
        moved_from = before.moved_from if before else None
        if before and before.slot_id != slot.id:
            moved_from = before.when
        self.changing = False
        self.pending = None
        self.appointment = Appointment(
            confirmation=(
                before.confirmation if before else f"RO-{self._ro_number + self._bookings - 1}"
            ),
            vin=v.vin,
            vehicle=v.label,
            services=[MENU[s].label for s in self.draft.services],
            concern=self.draft.concern,
            recall_id=recall.id if recall else None,
            recall=self.draft.recall if recall else None,
            slot_id=slot.id,
            when=slot.spoken(),
            advisor=slot.advisor,
            transport=self.draft.transport,
            hours=self.visit_hours(),
            estimate=self.estimate(),
            notes=list(self.draft.notes),
            moved_from=moved_from,
            approval_limit=self.draft.approval_limit,
            stop_driving=self.stop_driving(),
        )
        return self.appointment

    # ---- what's already booked: the appointment on file, or the one booked on this call ----

    def booked(self, which: Booking) -> Appointment:
        """A booked appointment that can still be moved or cancelled."""
        self._require_customer()
        a = self.on_file if which == "existing" else self.appointment
        if a is None:
            raise ServiceError(
                "Nothing's been booked on this call yet: for the visit you're putting together, "
                "set the time with choose_slot."
                if which == "new"
                else "There's no other appointment on their file."
            )
        if a.status == "cancelled":
            raise ServiceError(f"The {a.what} on {a.when} is already cancelled.")
        return a

    def change_booking(self) -> Appointment:
        """Opens the booking made on this call for changes: a service, the car, the ride. It's
        booked again, under the same number, after a new read-back."""
        a = self.booked("new")
        self.changing = True
        return a

    def propose_move(self, which: Booking, slot_id: str) -> str:
        """The move, as the agent reads it back before making it; nothing changes yet."""
        a = self.booked(which)
        slot = self._move_slot(a, slot_id)
        self.pending = Pending("move", which, slot.id)
        return (
            f"Move the {a.what} from {a.when} to {slot.spoken()}, with {slot.advisor}? It's "
            "the same visit otherwise."
        )

    def move(self, which: Booking, slot_id: str) -> Appointment:
        """Makes a move the caller agreed to, once it's been read back."""
        a = self.booked(which)
        slot = self._move_slot(a, slot_id)
        if self.pending != Pending("move", which, slot.id):
            raise ServiceError(
                "Read the move back first: call move_appointment without caller_confirmed, say "
                "what it returns, and wait for a clear yes."
            )
        self._slot(a.slot_id).open = True
        slot.open = False
        a.moved_from, a.slot_id, a.when, a.advisor = a.when, slot.id, slot.spoken(), slot.advisor
        if which == "new" and not self.changing:
            # the draft is what was booked: keep it in step, already read back
            self.draft.slot_id = slot.id
            self._changed()
            self.draft.read_back_revision = self.draft.revision
        self.pending = None
        return a

    def propose_cancel(self, which: Booking) -> str:
        a = self.booked(which)
        self.pending = Pending("cancel", which)
        return f"Cancel the {a.what} on {a.when}? Nothing is charged for cancelling."

    def cancel(self, which: Booking) -> Appointment:
        """Cancels an appointment the caller agreed to cancel, once it's been read back."""
        a = self.booked(which)
        if self.pending != Pending("cancel", which):
            raise ServiceError(
                "Read the cancellation back first: call cancel_appointment without "
                "caller_confirmed, say what it returns, and wait for a clear yes."
            )
        self._slot(a.slot_id).open = True
        a.status = "cancelled"
        self.pending = None
        if which == "new":
            # booking the visit again is a new booking, read back again first
            self.changing = False
            self._changed()
        return a

    def duplicates(self) -> list[str]:
        """Services on the draft that the appointment on file already has for the same car."""
        e = self.on_file
        if e is None or e.status != "booked" or e.vin != self.draft.vin:
            return []
        return [MENU[s].label for s in self.draft.services if MENU[s].label in e.services]

    def _duplicate_reason(self) -> str:
        e = self.on_file
        assert e is not None
        dup = " and ".join(self.duplicates()).lower()
        return (
            f"They already have the {e.what} booked for {e.when}, the appointment on their "
            f"file. Never book it twice: tell them, and offer to move that appointment "
            f"(move_appointment) instead, or leave the {dup} off this visit."
        )

    def _move_slot(self, a: Appointment, slot_id: str) -> Slot:
        if a is self.appointment and self.changing:
            raise ServiceError(
                "This call's booking is open for changes: set the new time with choose_slot, "
                "then read_back and book_service."
            )
        slot = self._slot(slot_id)
        if slot.id == a.slot_id:
            raise ServiceError(f"The {a.what} is already booked for {a.when}.")
        if not slot.open:
            raise ServiceError(f"{slot.spoken()} is taken. Offer another time.")
        if not slot.fits(a.hours):
            raise ServiceError(
                f"A visit of {spoken_span(a.hours)} won't finish by closing if it starts at "
                f"{clock(slot.start)}. Offer an earlier time."
            )
        return slot

    def _free(self, slot: Slot) -> bool:
        """Open, or the time the booking being changed already holds."""
        mine = self.appointment.slot_id if self.changing and self.appointment else None
        return slot.open or slot.id == mine

    # ---- a car in the shop (the follow-up call) ----

    def confirm_customer(self) -> ShopVisit:
        """Whoever picked up said they're the customer: the visit can be talked about now."""
        v = self._shop()
        if v.reached == "someone_else":
            who = self._first_name()
            raise ServiceError(
                f"Whoever answered already said they aren't {who}, so share nothing about the "
                f"car or the visit. Ask that {who} call back, say goodbye, and call end_call."
            )
        v.reached = "customer"
        v.ended_early = None
        return v

    def repair_details(self, question: str = "") -> list[Finding]:
        v = self._with_customer()
        if asked := " ".join(question.split()):
            v.questions.append(asked)
        return list(v.findings)

    def finding(self, query: str) -> Finding:
        """A finding by its id, or by words like "front" or "the rotors"."""
        v = self._shop()
        q = query.strip().lower()
        for f in v.findings:
            if q == f.id:
                return f
        words = [w for w in re.findall(r"[a-z]+", q) if w not in {"the", "brake", "brakes", "s"}]
        hits = [f for f in v.findings if words and all(w in f.title.lower() for w in words)]
        if len(hits) == 1:
            return hits[0]
        found = ", ".join(f"{f.id} ({f.title.lower()})" for f in v.findings)
        raise ServiceError(f"Couldn't tell which repair {query!r} is. The findings: {found}.")

    def decide_repair(self, query: str, decision: RepairDecision) -> tuple[Finding, list[str]]:
        """Approve, decline, or think over one repair; returns it and what moved because of it."""
        v = self._with_customer()
        f = self.finding(query)
        if decision not in DECISIONS:
            raise ServiceError("A repair is approved, declined, or deferred to think it over.")
        notes: list[str] = []
        if v.decisions.get(f.id) != decision:
            ready, shuttle = self.ready_when(), self.shuttle_when()
            v.decisions[f.id] = decision
            self._visit_changed()
            if self.ready_when() != ready:
                notes.append(f"The car is now ready {self.ready_when()}.")
            if v.ride == "shuttle" and self.shuttle_when() != shuttle:
                notes.append(
                    f"The shuttle now picks them up {self.shuttle_when()}, to match. Tell them."
                )
        return f, notes

    def note_for_next_visit(self, note: str) -> str:
        """Something to do at the next visit, on the customer's file."""
        v = self._with_customer()
        note = " ".join(note.split())
        if not note:
            raise ServiceError('Pass the note in a few words, e.g. "rear brake pads".')
        if note.lower() not in (n.lower() for n in v.next_visit):
            v.next_visit.append(note)
            self._visit_changed()
        return note

    def arrange_pickup(self, ride: str, where: str = "", collected_by: str = "") -> ShopVisit:
        v = self._with_customer()
        collected_by = " ".join(collected_by.split())
        if collected_by != v.collected_by:
            v.collected_by = collected_by
            self._visit_changed()
        if ride not in RIDES:
            raise ServiceError(
                "The options are the free courtesy shuttle, or coming in themselves."
            )
        where = " ".join(where.split()) if ride == "shuttle" else ""
        if ride == "shuttle" and not where:
            raise ServiceError(
                f"Ask where the shuttle should pick them up (home or work, within {SHUTTLE_MILES} "
                "miles), then call arrange_pickup again with it."
            )
        if (v.ride, v.pickup_from) != (ride, where):
            v.ride = ride
            v.pickup_from = where
            self._visit_changed()
        return v

    def pickup_summary(self) -> str:
        """When the car's ready, what's due, and the ride, as the agent must say it."""
        v = self._with_customer()
        if open_ := self.undecided():
            f = open_[0]
            raise ServiceError(
                f"They haven't decided on the {f.title.lower()} yet. Ask whether to go ahead "
                "today. If they want to think it over, call decline_repair with think_it_over "
                "true."
            )
        if v.ride is None:
            raise ServiceError(
                "Ask how they'll get back for the car: the free courtesy shuttle, or on their "
                "own. Then call arrange_pickup."
            )
        v.summary_rev = v.rev
        return self.ready_summary()

    def note_concern(self, concern: str) -> str:
        """Something new the customer has noticed, on the visit for the technician to check
        when they come in. Nothing is admitted or promised by noting it."""
        v = self._with_customer()
        concern = " ".join(concern.split())
        if not concern:
            raise ServiceError('Pass what they noticed in a few words, e.g. "new rattle".')
        if concern.lower() not in (c.lower() for c in v.concerns):
            v.concerns.append(concern)
        return concern

    def end_without_update(self, reason: EndedEarly) -> ShopVisit:
        """Wrong person, or a bad time: the car stays as it is, nothing decided."""
        v = self._shop()
        if reason not in ENDED_EARLY:
            raise ServiceError("The reason is wrong_person or call_back_later.")
        if v.reached == "customer" and v.decisions:
            raise ServiceError(
                "You've already gone over the repairs with them, so finish the call: set the "
                "ride with arrange_pickup if it isn't set, call pickup_summary, and say what it "
                "returns. Then end_call."
            )
        if reason == "wrong_person":
            v.reached = "someone_else"
        v.ended_early = reason
        return v

    def undecided(self) -> list[Finding]:
        """The repairs that need doing now and have no answer yet."""
        v = self._shop()
        return [f for f in v.findings if f.urgency == "now" and f.id not in v.decisions]

    def approved(self) -> list[Finding]:
        v = self._shop()
        return [f for f in v.findings if v.decisions.get(f.id) == "approved"]

    def ready_at(self, repairs: list[Finding] | None = None) -> dt.datetime | None:
        """When the car's ready with these repairs (the approved ones, by default): None means
        now, since the diagnostic and the recall are done. Work that fits the afternoon is done
        by closing, longer work the next business morning."""
        hours = sum(f.labor_hours for f in (self.approved() if repairs is None else repairs))
        if not hours:
            return None
        if hours <= SAME_DAY_HOURS:
            return dt.datetime.combine(self.today, SHOP_CLOSES)
        return dt.datetime.combine(_business_days(self.today, 1)[0], NEXT_DAY_READY)

    def ready_when(self, repairs: list[Finding] | None = None) -> str:
        ready = self.ready_at(repairs)
        return f"{self._day_word(ready)} at {clock(ready)}" if ready else "now"

    def shuttle_when(self) -> str:
        ready = self.ready_at()
        if ready is None:
            return "in about half an hour"
        return f"{self._day_word(ready)} at {clock(ready - SHUTTLE_LEAD)}"

    def ticket(self) -> list[tuple[str, float]]:
        """The repair order's lines and prices, with each repair once it's approved."""
        v = self._shop()
        price = {m.label: m.price for m in MENU.values()}
        lines = [(label, price[label]) for label in v.appointment.services]
        if v.appointment.recall == "added":
            lines.append((f"{self._recall_title()} recall", 0.0))
        lines += [(f.title, f.price) for f in self.approved()]
        return lines

    def ticket_total(self) -> float:
        return sum(price for _, price in self.ticket())

    def ready_summary(self) -> str:
        v = self._shop()
        lines = ", ".join(
            f"{label.lower()} " + (f"${price:.0f}" if price else "no charge")
            for label, price in self.ticket()
        )
        done = " and ".join(f.title.lower() for f in self.approved())
        parts = [
            f"The {self._shop_car().name.split()[0]} is ready {self.ready_when()}"
            + (f", once the {done} are done." if done else "."),
            f"Due at pickup: ${self.ticket_total():.0f} ({lines}), paid in person at the service "
            "desk when they collect the car; nothing is paid over the phone.",
        ]
        if v.ride == "shuttle":
            parts.append(
                f"The courtesy shuttle picks them up from {v.pickup_from} {self.shuttle_when()}, "
                "no charge."
            )
        elif v.ride == "self":
            closes = clock(dt.datetime.combine(self.today, SHOP_CLOSES))
            parts.append(f"They're coming in themselves; the shop's open until {closes}.")
        if v.collected_by:
            parts.append(
                f"{v.collected_by} is collecting it: they give {self._first_name()}'s name or "
                f"the repair order number at the service desk, and pay there."
            )
        if v.concerns:
            parts.append(
                f"Noted for the technician to check when they come in: {'; '.join(v.concerns)}."
            )
        for f in v.findings:
            d = v.decisions.get(f.id)
            if d == "deferred":
                parts.append(
                    f"The {f.title.lower()} are noted on the file while they think it over."
                )
            elif d == "declined" and f.urgency == "now":
                parts.append(f"The {f.title.lower()} are noted as declined for today.")
        if v.next_visit:
            parts.append(f"For the next visit: {'; '.join(v.next_visit)}.")
        return " ".join(parts)

    @property
    def call_done(self) -> bool:
        """The customer heard the ready time and total, or the call ended without the update."""
        v = self.visit
        return v is not None and (v.ended_early is not None or v.summary_rev == v.rev)

    def _day_word(self, when: dt.datetime) -> str:
        if when.date() == self.today:
            return "today"
        if when.date() == self.today + dt.timedelta(days=1):
            return "tomorrow"
        return f"{when:%A}"

    def _shop(self) -> ShopVisit:
        if self.visit is None:
            raise ServiceError("There's no car in the shop on this call.")
        return self.visit

    def _with_customer(self) -> ShopVisit:
        """The visit, once whoever picked up has said they're the customer."""
        v = self._shop()
        who = self._first_name()
        if v.reached == "someone_else":
            raise ServiceError(
                f"Whoever answered isn't {who}: share nothing about the car, the visit, or the "
                f"repairs. Ask them to have {who} call back, say goodbye, and call end_call."
            )
        if v.reached == "unknown":
            raise ServiceError(
                f"You don't know it's {who} yet. Once they've said it's them, call "
                "confirm_customer; until then, say nothing about the car."
            )
        return v

    def _visit_changed(self) -> None:
        self._shop().rev += 1

    def _first_name(self) -> str:
        return first_name(self.customer.name if self.customer else "")

    def _shop_car(self) -> Vehicle:
        return self._vehicle_by_vin(self._shop().appointment.vin)

    def _recall_title(self) -> str:
        car = self._shop_car()
        return car.recalls[0].title if car.recalls else "Open"

    # ---- derived ----

    def draft_vehicle(self) -> Vehicle | None:
        return self._vehicle_by_vin(self.draft.vin) if self.draft.vin else None

    def open_recall(self) -> Recall | None:
        v = self.draft_vehicle()
        if not v or v.vin not in self.recalls_checked or not v.recalls:
            return None
        return v.recalls[0]

    def visit_hours(self) -> float:
        hours = sum(MENU[s].hours for s in self.draft.services)
        recall = self.open_recall()
        if recall and self.draft.recall == "added":
            hours += recall.hours
        return hours

    def estimate(self) -> float:
        return sum(MENU[s].price for s in self.draft.services)

    def loaner_eligible(self) -> bool:
        covered = "warranty_repair" in self.draft.services or (
            self.draft.recall == "added" and self.open_recall() is not None
        )
        return covered and self.visit_hours() > LOANER_MIN_HOURS

    def missing(self) -> list[str]:
        v = self.draft_vehicle()
        out: list[str] = []
        if self.customer is None:
            out.append("the customer")
        if v is None:
            out.append("the vehicle")
        if not self.draft.services:
            out.append("the service")
        if v is not None and v.vin not in self.recalls_checked:
            out.append("a recall check")
        elif v is not None and v.recalls and self.draft.recall is None:
            out.append("whether to add the recall")
        if self.draft.slot_id is None:
            out.append("a time")
        if self.draft.transport is None:
            out.append("transportation")
        return out

    def stage(self) -> str:
        if self.visit is not None:
            return self._visit_stage()
        v = self.draft_vehicle()
        if self.appointment and self.appointment.status == "booked" and not self.changing:
            return "booked"
        if self.customer is None:
            return "customer"
        if v is None:
            return "vehicle"
        if not self.draft.services:
            # the problem first, then the service that fits it
            return "service" if self.draft.concern else "symptoms"
        if v.vin not in self.recalls_checked or (v.recalls and self.draft.recall is None):
            return "recalls"
        if self.draft.slot_id is None or self.draft.transport is None:
            return "time"
        return "readback"

    def _visit_stage(self) -> str:
        v = self._shop()
        if v.reached != "customer":
            return "reach"
        if self.undecided():
            return "quote"
        if v.summary_rev != v.rev:
            return "pickup"
        return "wrapup"

    def summary(self) -> str:
        v = self.require_vehicle()
        slot = self._slot(self.draft.slot_id or "")
        concern = f" for {self.draft.concern}" if self.draft.concern else ""
        services = [
            MENU[s].label.lower() + (concern if MENU[s].kind != "menu" else "")
            for s in self.draft.services
        ]
        parts = [f"{v.label}", ", ".join(services)]
        if self.draft.concern and all(MENU[s].kind == "menu" for s in self.draft.services):
            parts.append(f"noted for the technician: {self.draft.concern}")
        recall = self.open_recall()
        if recall:
            if self.draft.recall == "added":
                parts.append(f"plus the {recall.title.lower()} recall, no charge")
            else:
                parts.append("recall declined for now")
        parts.append(f"{slot.spoken()} ({self.day_word(slot.start.date())}) with {slot.advisor}")
        parts.append(TRANSPORT_LABELS[self.draft.transport or "shuttle"].lower())
        parts.append(f"estimate ${self.estimate():.0f}" if self.estimate() else "no charge")
        if (limit := self.draft.approval_limit) is not None:
            parts.append(f"we'll call before doing any work over ${limit:.0f}")
        out = "; ".join(parts) + "."
        if why := self.stop_driving():
            out += (
                f" It isn't safe to drive ({why}): have it towed in, not driven; the desk can't "
                "arrange the tow."
            )
        return out

    # ---- what the page shows ----

    def snapshot(self) -> dict[str, Any]:
        c = self.customer
        v = self.draft_vehicle()
        slot = next((s for s in self.slots if s.id == self.draft.slot_id), None)
        recall = self.open_recall()
        lines: list[dict[str, Any]] = []
        for s in self.draft.services:
            item = MENU[s]
            note = {"menu": "", "fee": "Fee only · repair quoted after", "warranty": "Covered"}
            lines.append(
                {
                    "id": s,
                    "label": item.label,
                    "price": item.price,
                    "hours": item.hours,
                    "note": note[item.kind],
                }
            )
        return {
            "today": self.today.isoformat(),
            # isNew: a file set up on this call, for a name and number that had none
            "customer": {"name": c.name, "phone": _phone(c.phone), "isNew": c.new} if c else None,
            "vehicles": [self._vehicle_dict(x) for x in c.vehicles] if c else [],
            "menu": [
                {"id": m.id, "label": m.label, "price": m.price, "hours": m.hours, "kind": m.kind}
                for m in MENU.values()
            ],
            "slots": [
                {
                    "id": s.id,
                    "day": f"{s.start:%a}",
                    "date": f"{s.start:%b} {s.start.day}",
                    "time": clock(s.start),
                    "advisor": s.advisor,
                    "open": s.open,
                    "fits": s.fits(self.visit_hours()),
                }
                for s in self.slots
            ],
            "quotes": [
                {"asked": q.asked, "price": q.price, "answer": q.answer} for q in self.quotes
            ],
            "draft": {
                "vin": v.vin if v else None,
                "vehicle": v.label if v else None,
                "mileage": v.miles if v else None,
                "services": lines,
                "concern": self.draft.concern,
                "notes": [_note_dict(n) for n in self.draft.notes],
                "likelyCause": _cause_dict(self.likely_cause()),
                "recall": (
                    {
                        "id": recall.id,
                        "title": recall.title,
                        "hours": recall.hours,
                        "decision": self.draft.recall,
                    }
                    if recall
                    else None
                ),
                "recallChecked": bool(v and v.vin in self.recalls_checked),
                "slot": (
                    {"id": slot.id, "when": slot.spoken(), "advisor": slot.advisor}
                    if slot
                    else None
                ),
                "transport": self.draft.transport,
                "approvalLimit": self.draft.approval_limit,
                # why it shouldn't be driven in, from what the caller said, if it shouldn't
                "stopDriving": self.stop_driving(),
                "transportLabel": TRANSPORT_LABELS.get(self.draft.transport or ""),
                "loanerEligible": self.loaner_eligible(),
                "hours": self.visit_hours(),
                "estimate": self.estimate(),
                "missing": self.missing(),
                # the booking made on this call is open for changes, to be booked again
                "changingBooking": self.changing,
                "revision": self.draft.revision,
                "readBack": self.draft.read_back_revision == self.draft.revision
                and self.draft.read_back_revision is not None,
            },
            "appointment": _appointment_dict(self.appointment),
            # the appointment already on their file, once they're looked up
            "existingAppointment": _appointment_dict(self.on_file) if c else None,
            # a move or cancellation read back and waiting for the caller's yes
            "pendingChange": self._pending_dict(),
            "visit": self._visit_dict(),
        }

    def _pending_dict(self) -> dict[str, Any] | None:
        p = self.pending
        if p is None:
            return None
        a = self.on_file if p.booking == "existing" else self.appointment
        to = self._slot(p.slot_id) if p.slot_id else None
        return {
            "action": p.action,
            "appointment": p.booking,
            "confirmation": a.confirmation if a else None,
            "to": to.spoken() if to else None,
        }

    def _visit_dict(self) -> dict[str, Any] | None:
        v = self.visit
        if v is None:
            return None
        a = v.appointment
        ready = self.ready_at()
        when = f"{self._day_word(ready).capitalize()}, " if ready else ""
        return {
            "confirmation": a.confirmation,
            "vehicle": a.vehicle,
            "advisor": a.advisor,
            "droppedOff": clock(dt.datetime.combine(self.today, DROPPED_OFF)),
            "concern": a.concern,
            "notes": [_note_dict(n) for n in a.notes],
            "reached": v.reached,
            "endedEarly": ENDED_EARLY[v.ended_early] if v.ended_early else None,
            # what the technician found stays off the page until the customer is on the line
            "findings": [
                {
                    "id": f.id,
                    "title": f.title,
                    "found": f.found,
                    "partsDetail": f.parts_detail,
                    "parts": f.parts,
                    "labor": f.labor,
                    "laborHours": f.labor_hours,
                    "price": f.price,
                    "urgency": f.urgency,
                    "urgencyLabel": URGENCY[f.urgency],
                    "soon": f.soon,
                    "decision": v.decisions.get(f.id),
                    "decisionLabel": DECISIONS[d] if (d := v.decisions.get(f.id)) else None,
                }
                for f in v.findings
            ]
            if v.reached == "customer"
            else [],
            "nextVisit": list(v.next_visit),
            "collectedBy": v.collected_by or None,
            "concerns": list(v.concerns),
            "questions": list(v.questions),
            "ticket": [{"label": label, "price": price} for label, price in self.ticket()],
            "total": self.ticket_total(),
            "readyAt": f"{when}{clock(ready)}" if ready else None,
            "ride": v.ride,
            "rideLabel": RIDES[v.ride] if v.ride else None,
            "pickupFrom": v.pickup_from or None,
            "shuttleAt": (
                (f"{when}{clock(ready - SHUTTLE_LEAD)}" if ready else "In about 30 min")
                if v.ride == "shuttle"
                else None
            ),
            "summaryRead": v.summary_rev == v.rev,
            "rev": v.rev,
        }

    def _vehicle_dict(self, v: Vehicle) -> dict[str, Any]:
        checked = v.vin in self.recalls_checked
        return {
            "vin": v.vin,
            "year": v.year,
            "name": v.name,
            "label": v.label,
            "body": v.body,
            "color": v.color,
            "miles": v.miles,
            "milesOnFile": v.miles_on_file,
            "inWarranty": v.in_warranty(self.today),
            "warranty": v.warranty_reason(self.today),
            "recallsChecked": checked,
            # recalls stay unknown on the page until the agent actually checks
            "recalls": [{"id": r.id, "title": r.title, "hours": r.hours} for r in v.recalls]
            if checked
            else None,
            "selected": v.vin == self.draft.vin,
            # put on the file on this call, from what the caller said
            "added": v.added,
        }

    # ---- internals ----

    def _changed(self) -> None:
        self.draft.revision += 1

    def _revalidate(self) -> list[str]:
        """Drop whatever a change just made invalid, and say so for the agent to relay."""
        notes: list[str] = []
        v = self.draft_vehicle()
        if v and "warranty_repair" in self.draft.services and not v.in_warranty(self.today):
            self.draft.services.remove("warranty_repair")
            self._changed()
            notes.append(
                f"Warranty repair removed: the {v.name} is out of warranty. Offer a diagnostic."
            )
        if self.draft.transport == "loaner" and not self.loaner_eligible():
            self.draft.transport = None
            self._changed()
            notes.append("Loaner removed: this visit no longer qualifies. Offer the shuttle.")
        slot = next((s for s in self.slots if s.id == self.draft.slot_id), None)
        if slot and not slot.fits(self.visit_hours()):
            self.draft.slot_id = None
            self._changed()
            notes.append(f"{slot.spoken()} no longer fits a visit this long. Pick another time.")
        return notes

    def _no_loaner_reason(self) -> str:
        return (
            f"No loaner: loaners are only for warranty or recall visits over "
            f"{LOANER_MIN_HOURS:g} hours, and this visit is {spoken_span(self.visit_hours())}. "
            "Offer the free shuttle, or waiting in the lounge."
        )

    def _require_customer(self) -> Customer:
        if self.customer is None:
            raise ServiceError("Look the customer up by phone number first.")
        return self.customer

    def _require_unbooked(self) -> None:
        # notes and the concern can come before the lookup: people start with the problem
        a = self.appointment
        if a and a.status == "booked" and not self.changing:
            raise ServiceError(
                f"Already booked as {a.confirmation}. To add to it or change it, call "
                "change_booking first; to move it or cancel it, use move_appointment or "
                "cancel_appointment."
            )

    def _require_open(self) -> None:
        self._require_customer()
        self._require_unbooked()

    def require_vehicle(self) -> Vehicle:
        v = self.draft_vehicle()
        if v is None:
            raise ServiceError("Find out which car it is first.")
        return v

    def _vehicle_by_vin(self, vin: str) -> Vehicle:
        c = self._require_customer()
        want = vin.strip().upper()
        for v in c.vehicles:
            if v.vin == want or (len(want) >= 6 and v.vin.endswith(want)):
                return v
        raise ServiceError(f"No vehicle with VIN {vin} on this customer's file.")

    def _slot(self, slot_id: str) -> Slot:
        for s in self.slots:
            if s.id == slot_id.strip().lower():
                return s
        raise ServiceError(f"No slot {slot_id!r}. Use an id from get_slots.")


def _score(v: Vehicle, words: list[str]) -> int:
    # "x3", "330", "2019", "the white one", "suv": words that start one of the car's own
    hay = f"{v.year} {v.name} {v.body} {v.color}".lower().split()
    return sum(
        1 for w in words if (len(w) >= 2 or w.isdigit()) and any(h.startswith(w) for h in hay)
    )


def _family(v: Vehicle) -> str:
    found = bmw_model(v.name)
    return found[1] if found else ""


_NUMBER_WORDS = dict(
    zip("one two three four five six seven eight".split(), "12345678", strict=True)
)
_OTHER_MAKES = re.compile(
    r"\b(toyota|honda|ford|chevy|chevrolet|tesla|audi|mercedes|benz|lexus|nissan|kia|hyundai|"
    r"mazda|subaru|volkswagen|vw|volvo|porsche|jeep|dodge|ram|gmc|cadillac|acura|infiniti|"
    r"lincoln|buick|mini|jaguar|land rover|range rover|genesis|mitsubishi|fiat|alfa romeo|"
    r"chrysler|rivian|lucid|polestar|tacoma|camry|civic|accord|corolla|prius|mustang)\b"
)
# trims after the model: xDrive40i, sDrive30i, eDrive40, M50i, M, Competition
_TRIM = re.compile(r"(?:([xse])drive(\d{2}[ie]?)?|m(\d{2}[ied]?)|m|competition)")


def bmw_model(said: str) -> tuple[str, str] | None:
    """A BMW model the desk services, as (its name, its family), from what the caller said:
    "x five" is ("X5", "X5"), "2019 330i" ("330i", "3 Series"), "X3 M40i" ("X3 M40i", "X3"),
    "3 series" ("3 Series", "3 Series"), "i4" ("i4", "i4"). None for anything else."""
    text = said.lower().replace("-", " ")
    if _OTHER_MAKES.search(text):
        return None
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [_NUMBER_WORDS.get(w, w) for w in text.split() if w != "bmw"]
    # "x 5", "i 4", "330 i": one word, the way it's badged
    text = re.sub(r"\b([xizm]) (\d)\b", r"\1\2", " ".join(words))
    text = re.sub(r"\b([1-8][1-6]\d) ([ied]|xi)\b", r"\1\2", text)
    words = text.split()
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if m := re.fullmatch(r"x([1-7])", w):
            name, family = f"X{m[1]}", f"X{m[1]}"
        elif w in {"xm", "z4"}:
            name = family = w.upper()
        elif m := re.fullmatch(r"ix([1-3])?", w):
            name = family = f"iX{m[1] or ''}"
        elif m := re.fullmatch(r"i([3-8])", w):
            name = family = f"i{m[1]}"
        elif m := re.fullmatch(r"m([2-8])", w):
            name = family = f"M{m[1]}"
        elif m := re.fullmatch(r"(m?)([1-8])([1-6]\d)(i|e|d|xi)", w):
            name, family = f"{m[1].upper()}{m[2]}{m[3]}{m[4]}", f"{m[2]} Series"
        elif re.fullmatch(r"[1-8]", w) and nxt in {"series", "serie"}:
            name = family = f"{w} Series"
            i += 1
        else:
            continue
        rest = words[i + 1 :]
        for j, t in enumerate(rest):
            # "M Sport" is a package, not the M model
            if not (m := _TRIM.fullmatch(t)) or rest[j + 1 : j + 2] == ["sport"]:
                break
            if m[1]:
                name += f" {m[1]}Drive{m[2] or ''}"
            elif m[3]:
                name += f" M{m[3]}"
            else:
                name += " M" if t == "m" else " Competition"
        return name, family
    return None


def spoken_span(amount: float, unit: str = "hour") -> str:
    """A length the way people say it: "half an hour", "an hour and a half", "4 hours"."""
    whole, half = int(amount), amount % 1 >= 0.5
    a = "an" if unit == "hour" else "a"
    if whole == 0:
        return f"half {a} {unit}"
    if whole == 1:
        return f"{a} {unit} and a half" if half else f"{a} {unit}"
    return f"{whole} and a half {unit}s" if half else f"{whole} {unit}s"


def clock(t: dt.datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def _phone(digits: str) -> str:
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"


def _appointment_dict(a: Appointment | None) -> dict[str, Any] | None:
    if a is None:
        return None
    return {
        "confirmation": a.confirmation,
        "vin": a.vin,
        "vehicle": a.vehicle,
        "services": a.services,
        "concern": a.concern,
        "recallId": a.recall_id,
        "recall": a.recall,
        "slotId": a.slot_id,
        "when": a.when,
        "advisor": a.advisor,
        "transport": a.transport,
        "transportLabel": TRANSPORT_LABELS[a.transport],
        "hours": a.hours,
        "estimate": a.estimate,
        "notes": [_note_dict(n) for n in a.notes],
        "status": a.status,
        "movedFrom": a.moved_from,
        "approvalLimit": a.approval_limit,
        "stopDriving": a.stop_driving,
    }


def _note_dict(n: TechNote) -> dict[str, str]:
    return {"kind": n.kind, "label": n.label, "text": n.text}


def _cause_dict(c: LikelyCause | None) -> dict[str, Any] | None:
    return {"cause": c.cause, "unsafe": c.unsafe} if c else None


def local_today(timezone: str | None = None) -> dt.date:
    """Today where the caller is: `timezone` is an IANA name (a browser's, e.g.
    "America/Toronto"); without one, or with one that's unknown, the dealership's today."""
    try:
        tz = ZoneInfo(timezone) if timezone else TIMEZONE
    except (ZoneInfoNotFoundError, ValueError):
        tz = TIMEZONE
    return dt.datetime.now(tz).date()


def same_person(said: str, on_file: str) -> bool:
    """The name a caller gave is the account's: the same first name, and the same last name
    if they gave one."""
    a, b = said.lower().split(), on_file.lower().split()
    if not a or not b or a[0] != b[0]:
        return False
    return len(a) == 1 or len(b) == 1 or a[-1] == b[-1]


def first_name(name: str) -> str:
    return name.split()[0] if name.strip() else "there"


def real_us_number(digits: str) -> bool:
    """Ten digits that could be a real North American number: an area code and exchange that
    don't start with 0 or 1 and aren't N11 codes (411, 911), and not one digit repeated."""
    if not re.fullmatch(r"\d{10}", digits) or len(set(digits)) == 1:
        return False
    area, exchange = digits[:3], digits[3:6]
    return all(p[0] not in "01" and p[1:] != "11" for p in (area, exchange))


def _quote_words(asked: str) -> list[str]:
    """What's left of a price question once filler, the car, and plurals are dropped."""
    words = re.sub(r"[^a-z0-9 ]", " ", asked.lower()).split()
    words = [w for w in words if w not in _QUOTE_FILLER and not re.search(r"\d", w)]
    return [w[:-1] if w.endswith("s") and not w.endswith(("ss", "is", "us")) else w for w in words]


def _menu_items(words: list[str]) -> list[MenuItem]:
    """Each priced item in "oil change and tire rotation", or nothing if any part isn't one."""
    parts: list[list[str]] = [[]]
    for w in words:
        if w in _QUOTE_JOINERS:
            parts.append([])
        else:
            parts[-1].append(w)
    items = [MENU.get(_QUOTE_ALIASES.get(" ".join(p), "")) for p in parts if p]
    if not items or any(i is None or i.kind == "warranty" for i in items):
        return []
    return list(dict.fromkeys(i for i in items if i is not None))


def _business_days(start: dt.date, n: int) -> list[dt.date]:
    days: list[dt.date] = []
    d = start
    while len(days) < n:
        d += dt.timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


# already taken: (weekday, slot index), so the same times are open whatever day it is. Never
# Tuesday at 8:00 or Wednesday's first slot, which the README and the tests ask for.
_BOOKED = {(0, 1), (2, 2), (3, 0), (3, 4), (4, 3)}
# the appointment already on Alex's file: never one of _BOOKED's times
UPCOMING_AT = dt.time(14, 30)


def demo_desk(
    customer_name: str = "Alex Morgan",
    today: dt.date | None = None,
    direction: Literal["inbound", "outbound"] = "inbound",
) -> ServiceDesk:
    """Alex Morgan's file: an in-warranty X3 with an open recall, and an older 330i. Outbound,
    the X3 is in the shop today, and the inspection found worn front brakes."""
    today = today or local_today()
    x3 = Vehicle(
        vin="WBX57DP0XPN123456",
        year=2023,
        name="X3 xDrive30i",
        body="suv",
        color="Brooklyn Grey",
        miles=21_400,
        # about two and a half years in service, so it stays in warranty whenever the demo runs
        in_service=today - dt.timedelta(days=940),
        recalls=recalls_for("X3", 2023),
    )
    sedan = Vehicle(
        vin="WBA5R1C05KA765432",
        year=2019,
        name="330i",
        body="sedan",
        color="Alpine White",
        miles=58_100,
        in_service=dt.date(2019, 6, 14),
    )
    for v in (x3, sedan):
        v.miles_on_file = v.miles
    customer = Customer(
        id="C-20417",
        name=" ".join(customer_name.split()) or "Alex Morgan",
        phone="4155550142",
        vehicles=[x3, sedan],
    )
    slots = [
        Slot(
            id=f"{day:%a}-{h:02d}{m:02d}".lower(),
            start=dt.datetime.combine(day, dt.time(h, m)),
            advisor=ADVISORS[(i + j) % len(ADVISORS)],
            open=(day.weekday(), j) not in _BOOKED,
        )
        for i, day in enumerate(_business_days(today, BOOKING_DAYS))
        for j, (h, m) in enumerate(SLOT_TIMES)
    ]
    desk = ServiceDesk([customer], slots, today, ro_number=48_210 + today.toordinal() % 700)
    if direction == "outbound":
        _seed_shop_visit(desk)
    else:
        _seed_upcoming(desk, sedan)
    return desk


def _seed_upcoming(desk: ServiceDesk, sedan: Vehicle) -> None:
    # booked on an earlier call: a tire rotation on the 330i at 2:30 on the calendar's last
    # day, which is always next week, and a time nothing else here asks for
    slot = next(s for s in reversed(desk.slots) if s.start.time() == UPCOMING_AT)
    slot.open = False
    rotation = MENU["tire_rotation"]
    desk._customers[0].on_file = Appointment(
        confirmation=f"RO-{desk._ro_number - 37}",
        vin=sedan.vin,
        vehicle=sedan.label,
        services=[rotation.label],
        concern="",
        recall_id=None,
        recall=None,
        slot_id=slot.id,
        when=slot.spoken(),
        advisor=slot.advisor,
        transport="wait",
        hours=rotation.hours,
        estimate=rotation.price,
    )


def _seed_shop_visit(desk: ServiceDesk) -> None:
    # booked earlier, the way the consultation call books it: a diagnostic for the grinding on
    # the X3, with the technician's notes, the open recall added, and the shuttle. It came in at
    # 7:30 this morning; the diagnostic and the recall are done.
    c = desk.lookup_customer(desk._customers[0].phone, desk._customers[0].name)
    x3 = c.vehicles[0]
    desk.select_vehicle(x3.vin)
    for kind, text in (
        ("sound", "Grinding, like metal on metal"),
        ("where", "Front, both sides"),
        ("when", "When braking, worse at low speed"),
        ("since", "About two weeks, getting louder"),
        ("warning_lights", "None"),
    ):
        desk.add_note(kind, text)
    desk.set_concern("Grinding from the front when braking")
    desk.set_services(["diagnostic"])
    desk.check_recalls(x3.vin)
    desk.decide_recall(True)
    desk.choose_transport("shuttle")
    desk.appointment = Appointment(
        confirmation=f"RO-{desk._ro_number}",
        vin=x3.vin,
        vehicle=x3.label,
        services=[MENU[s].label for s in desk.draft.services],
        concern=desk.draft.concern,
        recall_id=x3.recalls[0].id,
        recall="added",
        slot_id=f"today-{DROPPED_OFF:%H%M}",
        when=f"Today, {clock(dt.datetime.combine(desk.today, DROPPED_OFF))}",
        advisor=ADVISORS[0],
        transport="shuttle",
        hours=desk.visit_hours(),
        estimate=desk.estimate(),
        notes=list(desk.draft.notes),
    )
    desk.visit = ShopVisit(appointment=desk.appointment, findings=list(FINDINGS))
