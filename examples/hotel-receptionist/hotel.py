from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

RoomType = Literal["king", "queen", "suite"]
View = Literal["city", "ocean"]

TAX_RATE_PCT = 12
MAX_GUESTS = 4  # the largest room sleeps four
CANCELLATION_WINDOW_HOURS = 48
# HOTEL_TODAY pins the clock, so the scenarios' absolute dates stay where they are
TODAY = (
    date.fromisoformat(os.environ["HOTEL_TODAY"]) if os.environ.get("HOTEL_TODAY") else date.today()
)


@dataclass
class Room:
    number: str
    type: RoomType
    view: View
    nightly_rate: int  # dollars
    sleeps: int


@dataclass
class Booking:
    code: str
    room: Room
    check_in: date
    check_out: date
    guests: int
    first_name: str
    last_name: str
    email: str
    card_last4: str
    total: int  # dollars, tax included
    status: Literal["confirmed", "cancelled"] = "confirmed"

    @property
    def nights(self) -> int:
        return (self.check_out - self.check_in).days


class Unavailable(Exception):
    """No room of that kind is free for those dates."""


@dataclass
class Hotel:
    rooms: list[Room]
    bookings: list[Booking] = field(default_factory=list)

    @classmethod
    def seeded(cls) -> Hotel:
        hotel = cls(
            rooms=[
                Room("201", "king", "city", 240, 2),
                Room("202", "king", "ocean", 280, 2),
                Room("203", "queen", "city", 220, 4),
                Room("301", "queen", "ocean", 260, 4),
                Room("401", "suite", "ocean", 480, 4),
            ]
        )
        # the bookings the scenarios know: outside the cancellation window, inside it, one that
        # takes the only suite for a week, and two more to cancel, since scenarios run together
        seeded: list[tuple[RoomType, int, int, str, str, str, str]] = [
            ("king", 5, 7, "Eleanor", "Smith", "4242", "HTL-AB12"),
            ("queen", 1, 3, "Marcus", "Lee", "1881", "HTL-CD34"),
            ("suite", 18, 23, "Ana", "Garcia", "9001", "HTL-EF56"),
            ("king", 12, 14, "Priya", "Nair", "7007", "HTL-GH78"),
            ("queen", 26, 28, "Tom", "Okafor", "3310", "HTL-JK90"),
        ]
        for args in seeded:
            room_type, start, end, first_name, last_name, card_last4, code = args
            hotel.book(
                room_type=room_type,
                view=None,
                check_in=TODAY + timedelta(days=start),
                check_out=TODAY + timedelta(days=end),
                guests=2,
                first_name=first_name,
                last_name=last_name,
                email=f"{first_name.lower()}@example.com",
                card_last4=card_last4,
                code=code,
            )
        return hotel

    def available(
        self, *, check_in: date, check_out: date, guests: int, room_type: RoomType | None = None
    ) -> list[Room]:
        """The free rooms for the dates, one per type and view, cheapest first."""
        taken = {
            b.room.number
            for b in self.bookings
            if b.status == "confirmed" and b.check_in < check_out and b.check_out > check_in
        }
        free = [
            r
            for r in self.rooms
            if r.number not in taken
            and r.sleeps >= guests
            and (room_type is None or r.type == room_type)
        ]
        seen: dict[tuple[RoomType, View], Room] = {}
        for room in sorted(free, key=lambda r: r.nightly_rate):
            seen.setdefault((room.type, room.view), room)
        return list(seen.values())

    def quote(self, room: Room, check_in: date, check_out: date) -> int:
        """The stay's total with tax, in dollars."""
        nights = (check_out - check_in).days
        return round(room.nightly_rate * nights * (100 + TAX_RATE_PCT) / 100)

    def book(
        self,
        *,
        room_type: RoomType,
        view: View | None,
        check_in: date,
        check_out: date,
        guests: int,
        first_name: str,
        last_name: str,
        email: str,
        card_last4: str,
        code: str | None = None,
    ) -> Booking:
        rooms = [
            r
            for r in self.available(
                check_in=check_in, check_out=check_out, guests=guests, room_type=room_type
            )
            if view is None or r.view == view
        ]
        if not rooms:
            raise Unavailable
        alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
        booking = Booking(
            code=code or "HTL-" + "".join(secrets.choice(alphabet) for _ in range(4)),
            room=rooms[0],
            check_in=check_in,
            check_out=check_out,
            guests=guests,
            first_name=first_name,
            last_name=last_name,
            email=email,
            card_last4=card_last4,
            total=self.quote(rooms[0], check_in, check_out),
        )
        self.bookings.append(booking)
        return booking

    def find(self, *, last_name: str, code: str) -> Booking | None:
        """The booking under the name and the code, as a person says it: HTL-AB12, htl ab12,
        H-T-L-A-B-1-2."""
        letters = "".join(c for c in code if c.isalnum()).upper()
        return next(
            (
                b
                for b in self.bookings
                if b.code.replace("-", "") == letters and b.last_name.lower() == last_name.lower()
            ),
            None,
        )

    def refund(self, booking: Booking) -> int:
        """What a cancellation today gives back: everything, or less one night inside the window."""
        within = (booking.check_in - TODAY).days * 24 < CANCELLATION_WINDOW_HOURS
        return booking.total - (booking.room.nightly_rate if within else 0)

    def cancel(self, booking: Booking) -> int:
        """Cancels and returns the refund."""
        refund = self.refund(booking)
        booking.status = "cancelled"
        return refund
