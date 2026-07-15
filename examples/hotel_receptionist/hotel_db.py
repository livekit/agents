from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, time, timedelta
from itertools import groupby
from typing import get_args

import apsw

from livekit.agents.utils import shortuuid

from .hotel import (
    ALLOWED_EXTRAS,
    BUSINESS_CENTER_SERVICES,
    FLORIST_ARRANGEMENTS,
    SPA_SERVICES,
    TOURS,
    WALK_PARTNER_HOTEL,
    BusinessCenterService,
    ConflictResolution,
    DisputeCategory,
    EmailKind,
    EmergencyKind,
    FloralArrangement,
    FollowupKind,
    GroupShareType,
    Invoice,
    LineItem,
    NotFound,
    RestaurantReservation,
    RoomBooking,
    RoomExtra,
    RoomType,
    RoomTypeAvailability,
    SpaService,
    TimeSlot,
    Tour,
    TransferDestination,
    Unavailable,
    ViewRate,
    compute_invoice,
    normalize_email,
    normalize_phone,
)
from .hotel_schema import (
    _SQL_AVAILABILITY,
    _SQL_BOOKING_BY_CODE,
    _SQL_DINING_AVAILABILITY,
    _SQL_FIND_BOOKING,
    _SQL_FIND_RESERVATION,
    _SQL_FREE_BETTER_ROOM,
    _SQL_FREE_ROOM,
    _SQL_FREE_TABLE,
    _SQL_FREE_TABLE_FOR_MODIFY,
    _SQL_GET_INVOICE,
    _SQL_RESERVATION_BY_CODE,
    _SQL_ROOM_CONFLICT,
    _insert,
    _install_schema,
    _row_to_booking,
    _row_to_reservation,
    _update,
)

logger = logging.getLogger("hotel-receptionist.db")

OnChange = Callable[[], Awaitable[None]]


class HotelDB:
    def __init__(
        self, conn: apsw.Connection, today: date, *, on_change: OnChange | None = None
    ) -> None:
        self._conn: apsw.Connection = conn
        self.today = today
        self.on_change = on_change

    @classmethod
    def empty(cls, today: date, *, on_change: OnChange | None = None) -> HotelDB:
        conn = apsw.Connection(":memory:")
        _install_schema(conn, today)
        return cls(conn, today, on_change=on_change)

    @classmethod
    def from_bytes(
        cls, seed_bytes: bytes, today: date, *, on_change: OnChange | None = None
    ) -> HotelDB:
        conn = apsw.Connection(":memory:")
        conn.deserialize("main", seed_bytes)
        _install_schema(conn, today)
        return cls(conn, today, on_change=on_change)

    @classmethod
    def open_path(cls, db_path: str, today: date, *, on_change: OnChange | None = None) -> HotelDB:
        conn = apsw.Connection(db_path)
        _install_schema(conn, today)
        return cls(conn, today, on_change=on_change)

    @property
    def connection(self) -> apsw.Connection:
        return self._conn

    def serialize(self) -> bytes:
        return bytes(self._conn.serialize("main"))

    def close(self) -> None:
        self._conn.close()

    async def aclose(self) -> None:
        self.close()

    def _require_future(self, d: date) -> None:
        if d < self.today:
            raise Unavailable(f"{d.isoformat()} is in the past")

    async def _notify_change(self) -> None:
        if self.on_change:
            try:
                await self.on_change()
            except Exception:
                logger.exception("database mirror update failed")

    async def list_room_types_available(
        self,
        *,
        check_in: date,
        check_out: date,
        guests: int,
        smoking: bool | None = None,
        exclude_booking_code: str | None = None,
    ) -> list[RoomTypeAvailability]:
        rows = self.connection.execute(
            _SQL_AVAILABILITY,
            {
                "guests": guests,
                "smoking": int(smoking) if smoking is not None else None,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "exclude": exclude_booking_code,
            },
        )
        availability = []
        for room_type, group in groupby(rows, key=lambda row: row[0]):
            view_rates = tuple(
                ViewRate(view, minimum, maximum) for _, view, minimum, maximum in group
            )
            availability.append(RoomTypeAvailability(room_type, view_rates))
        return sorted(availability, key=lambda option: (option.lowest_nightly_rate, option.type))

    async def list_restaurant_availability(
        self, *, on_date: date, party_size: int
    ) -> list[TimeSlot]:
        rows = self.connection.execute(
            _SQL_DINING_AVAILABILITY, {"party_size": party_size, "date": on_date.isoformat()}
        )
        return [
            TimeSlot(time.fromisoformat(slot), [tid for _, tid in group if tid is not None])
            for slot, group in groupby(rows, key=lambda row: row[0])
        ]

    async def find_booking(
        self,
        *,
        last_name: str,
        confirmation_code: str | None = None,
        email: str | None = None,
        card_last4: str | None = None,
    ) -> RoomBooking | None:
        return _row_to_booking(
            self.connection.execute(
                _SQL_FIND_BOOKING,
                {
                    "last_name": last_name,
                    "code": confirmation_code.upper() if confirmation_code else None,
                    "email": email,
                    "card_last4": card_last4,
                },
            ).fetchone()
        )

    async def find_restaurant_reservation(
        self, *, last_name: str, confirmation_code: str | None = None, on_date: date | None = None
    ) -> RestaurantReservation | None:
        return _row_to_reservation(
            self.connection.execute(
                _SQL_FIND_RESERVATION,
                {
                    "last_name": last_name,
                    "code": confirmation_code.upper() if confirmation_code else None,
                    "date": on_date.isoformat() if on_date else None,
                },
            ).fetchone()
        )

    async def get_invoice(self, booking_code: str) -> Invoice:
        row = self.connection.execute(_SQL_GET_INVOICE, {"code": booking_code}).fetchone()
        if not row:
            raise NotFound(f"no invoice for {booking_code}")
        invoice_id, line_items_json, subtotal, taxes, total, paid = row
        return Invoice(
            id=invoice_id,
            booking_code=booking_code,
            line_items=[LineItem(**li) for li in json.loads(line_items_json)],
            subtotal=subtotal,
            taxes=taxes,
            total=total,
            paid=bool(paid),
        )

    async def book_room(
        self,
        *,
        room_type: RoomType,
        smoking: bool,
        guests: int,
        check_in: date,
        check_out: date,
        first_name: str,
        last_name: str,
        email: str,
        phone: str,
        card_last4: str,
        extras: list[RoomExtra],
        view: str | None = None,
    ) -> RoomBooking:
        clean_extras = sorted(e for e in extras if e in ALLOWED_EXTRAS)
        code = shortuuid("HTL-")
        conn = self.connection
        with conn:
            row = conn.execute(
                _SQL_FREE_ROOM,
                {
                    "room_type": room_type,
                    "smoking": int(smoking),
                    "guests": guests,
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "exclude": None,
                    "view": view,
                    "prefer": None,
                },
            ).fetchone()
            if not row:
                what = f"{view} {room_type}" if view else room_type
                raise Unavailable(f"sold out: {what}")
            room_id, nightly_rate = row
            nights = (check_out - check_in).days
            subtotal, taxes, total, items = compute_invoice(
                nightly_rate=nightly_rate, nights=nights, extras=clean_extras
            )
            booking_id = _insert(
                conn,
                "hotel_bookings",
                {
                    "code": code,
                    "room_id": room_id,
                    "first_name": first_name,
                    "last_name": last_name,
                    # spoken emails are case-free; transcription capitalization is noise
                    "email": normalize_email(email),
                    # digits only: a spoken number transcribes with unpredictable punctuation
                    "phone": normalize_phone(phone),
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "guests": guests,
                    "extras": ",".join(clean_extras),
                    "total": total,
                    "card_last4": card_last4,
                },
            )
            _insert(
                conn,
                "hotel_invoices",
                {
                    "booking_code": code,
                    "line_items": json.dumps([li.__dict__ for li in items]),
                    "subtotal": subtotal,
                    "taxes": taxes,
                    "total": total,
                },
            )
        await self._notify_change()
        return RoomBooking(
            id=booking_id,
            code=code,
            room_id=room_id,
            room_type=room_type,
            smoking=smoking,
            nightly_rate=nightly_rate,
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            check_in=check_in,
            check_out=check_out,
            guests=guests,
            extras=clean_extras,
            total=total,
            card_last4=card_last4,
            status="confirmed",
            late_arrival_note=None,
        )

    async def update_booking(
        self,
        *,
        booking_code: str,
        room_type: RoomType,
        smoking: bool,
        guests: int,
        check_in: date,
        check_out: date,
        extras: list[RoomExtra],
        view: str | None = None,
    ) -> RoomBooking:
        # Re-pick a free room of the new (type, smoking) for the new dates,
        # ignoring the booking being modified itself (so same-room "extend
        # by one night" doesn't conflict with itself). The room the guest
        # already has wins when it still fits - a date change must never
        # quietly move someone out of their garden view. A requested `view`
        # filters to rooms with that view, which is how the guest gets moved
        # to a different room (e.g. a city-view room to a garden-view one).
        clean_extras = sorted(e for e in extras if e in ALLOWED_EXTRAS)
        conn = self.connection
        with conn:
            current = conn.execute(
                "SELECT room_id FROM hotel_bookings WHERE code = ? AND status = 'confirmed'",
                (booking_code,),
            ).fetchone()
            row = conn.execute(
                _SQL_FREE_ROOM,
                {
                    "room_type": room_type,
                    "smoking": int(smoking),
                    "guests": guests,
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "exclude": booking_code,
                    "view": view,
                    "prefer": current[0] if current else None,
                },
            ).fetchone()
            if not row:
                what = f"{view} {room_type}" if view else room_type
                raise Unavailable(f"sold out: {what}")
            room_id, nightly_rate = row
            nights = (check_out - check_in).days
            subtotal, taxes, total, items = compute_invoice(
                nightly_rate=nightly_rate, nights=nights, extras=clean_extras
            )
            changed = _update(
                conn,
                "hotel_bookings",
                {
                    "room_id": room_id,
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "guests": guests,
                    "extras": ",".join(clean_extras),
                    "total": total,
                },
                {"code": booking_code, "status": "confirmed"},
            )
            if changed == 0:
                raise NotFound(f"booking not found: {booking_code}")
            _update(
                conn,
                "hotel_invoices",
                {
                    "line_items": json.dumps([li.__dict__ for li in items]),
                    "subtotal": subtotal,
                    "taxes": taxes,
                    "total": total,
                },
                {"booking_code": booking_code},
            )
        await self._notify_change()
        updated = _row_to_booking(
            conn.execute(_SQL_BOOKING_BY_CODE, {"code": booking_code}).fetchone()
        )
        if updated is None:
            raise NotFound(f"booking vanished mid-update: {booking_code}")
        return updated

    async def cancel_room_booking(self, booking_code: str) -> None:
        conn = self.connection
        changed = _update(
            conn,
            "hotel_bookings",
            {"status": "cancelled"},
            {"code": booking_code, "status": "confirmed"},
        )
        if changed == 0:
            raise NotFound(f"booking not found: {booking_code}")
        await self._notify_change()

    async def lookup_guest_history(self, *, last_name: str) -> str | None:
        """Return a returning guest's remembered preferences from past stays, or None
        if there's no history on file for that name."""
        row = self.connection.execute(
            "SELECT preferences FROM guest_history WHERE LOWER(last_name) = LOWER(?)",
            (last_name,),
        ).fetchone()
        return row[0] if row else None

    async def set_do_not_disturb(self, *, room: str) -> str:
        """Record a Do-Not-Disturb hold on a room and return a reference. The switchboard
        holds the room's calls and messages until it's lifted; emergencies override it."""
        # Normalize to the canonical room id (and reject a mis-heard room) so storage
        # matches every other room-referencing table regardless of how it was spoken.
        room_id = self._require_room(room)
        code = shortuuid("DND-")
        with self.connection as conn:
            _insert(conn, "do_not_disturb", {"code": code, "room_id": room_id})
        await self._notify_change()
        return code

    async def add_to_waitlist(
        self,
        *,
        first_name: str,
        last_name: str,
        phone: str,
        check_in: date,
        check_out: date,
        guests: int,
    ) -> str:
        """Record a waitlist entry for dates the hotel is sold out on, and return a
        reference. No room is held - the desk calls back only if something frees up."""
        code = shortuuid("WL-")
        with self.connection as conn:
            _insert(
                conn,
                "waitlist",
                {
                    "code": code,
                    "first_name": first_name,
                    "last_name": last_name,
                    "phone": normalize_phone(phone),
                    "check_in": check_in.isoformat(),
                    "check_out": check_out.isoformat(),
                    "guests": guests,
                },
            )
        await self._notify_change()
        return code

    async def reinstate_booking(self, booking_code: str) -> None:
        """Reactivate a previously cancelled booking, but only if its original room is
        still free for its dates. Raises NotFound if the code is unknown, Unavailable if
        the room has since been taken. A no-op if the booking is already confirmed."""
        conn = self.connection
        row = conn.execute(
            "SELECT room_id, check_in, check_out, status FROM hotel_bookings WHERE code = ?",
            (booking_code,),
        ).fetchone()
        if row is None:
            raise NotFound(f"booking not found: {booking_code}")
        room_id, check_in, check_out, status = row
        if status == "confirmed":
            return
        clash = conn.execute(
            "SELECT 1 FROM hotel_bookings WHERE room_id = ? AND status = 'confirmed' "
            "AND code != ? AND NOT (check_out <= ? OR check_in >= ?) LIMIT 1",
            (room_id, booking_code, check_in, check_out),
        ).fetchone()
        if clash:
            raise Unavailable("that room is no longer free for those dates")
        with conn:
            _update(conn, "hotel_bookings", {"status": "confirmed"}, {"code": booking_code})
        await self._notify_change()

    async def book_restaurant(
        self,
        *,
        first_name: str,
        last_name: str,
        phone: str,
        party_size: int,
        on_date: date,
        at_time: time,
        notes: str | None = None,
    ) -> RestaurantReservation:
        conn = self.connection
        row = conn.execute(
            _SQL_FREE_TABLE,
            {"party_size": party_size, "date": on_date.isoformat(), "time": at_time.isoformat()},
        ).fetchone()
        if not row:
            raise Unavailable(f"restaurant full: {on_date} {at_time}")
        table_id = row[0]
        code = shortuuid("RES-")
        reservation_id = _insert(
            conn,
            "restaurant_reservations",
            {
                "code": code,
                "table_id": table_id,
                "first_name": first_name,
                "last_name": last_name,
                "phone": normalize_phone(phone),
                "party_size": party_size,
                "date": on_date.isoformat(),
                "time": at_time.isoformat(),
                "notes": notes,
            },
        )
        await self._notify_change()
        return RestaurantReservation(
            id=reservation_id,
            code=code,
            table_id=table_id,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            party_size=party_size,
            date=on_date,
            time=at_time,
            notes=notes,
            status="confirmed",
        )

    async def cancel_restaurant_reservation(self, code: str) -> None:
        conn = self.connection
        changed = _update(
            conn,
            "restaurant_reservations",
            {"status": "cancelled"},
            {"code": code, "status": "confirmed"},
        )
        if changed == 0:
            raise NotFound(f"reservation not found: {code}")
        await self._notify_change()

    async def modify_restaurant_reservation(
        self,
        *,
        code: str,
        on_date: date,
        at_time: time,
        party_size: int | None = None,
    ) -> RestaurantReservation:
        """Change a confirmed reservation's date/time (and optionally party size).

        Prefers the reservation's current table when it's still free at the new
        slot (mirrors the prefer-current-room logic in update_booking), so a
        same-table time shift leaves table_location unchanged. Falls back to the
        next free table only if the current one is taken or too small.
        """
        self._require_future(on_date)
        conn = self.connection
        with conn:
            current = conn.execute(
                "SELECT table_id, party_size FROM restaurant_reservations "
                "WHERE code = :code AND status = 'confirmed'",
                {"code": code},
            ).fetchone()
            if not current:
                raise NotFound(f"reservation not found: {code}")
            current_table_id, current_party = current
            new_party = party_size if party_size is not None else current_party
            row = conn.execute(
                _SQL_FREE_TABLE_FOR_MODIFY,
                {
                    "party_size": new_party,
                    "date": on_date.isoformat(),
                    "time": at_time.isoformat(),
                    "code": code,
                    "current_table_id": current_table_id,
                },
            ).fetchone()
            if not row:
                raise Unavailable(f"restaurant full: {on_date} {at_time}")
            table_id = row[0]
            _update(
                conn,
                "restaurant_reservations",
                {
                    "table_id": table_id,
                    "party_size": new_party,
                    "date": on_date.isoformat(),
                    "time": at_time.isoformat(),
                },
                {"code": code, "status": "confirmed"},
            )
        await self._notify_change()
        updated = _row_to_reservation(
            conn.execute(_SQL_RESERVATION_BY_CODE, {"code": code}).fetchone()
        )
        if updated is None:
            raise NotFound(f"reservation vanished mid-update: {code}")
        return updated

    async def flag_late_arrival(self, *, booking_code: str, note: str) -> None:
        conn = self.connection
        changed = _update(
            conn,
            "hotel_bookings",
            {"late_arrival_note": note},
            {"code": booking_code, "status": "confirmed"},
        )
        if changed == 0:
            raise NotFound(f"booking not found: {booking_code}")
        await self._notify_change()

    async def update_booking_card(self, *, booking_code: str, card_last4: str) -> None:
        conn = self.connection
        changed = _update(
            conn,
            "hotel_bookings",
            {"card_last4": card_last4},
            {"code": booking_code, "status": "confirmed"},
        )
        if changed == 0:
            raise NotFound(f"booking not found: {booking_code}")
        await self._notify_change()

    async def record_followup(
        self,
        *,
        kind: FollowupKind,
        caller_name: str,
        caller_phone: str,
        summary: str,
    ) -> str:
        code = shortuuid("FUP-")
        conn = self.connection
        with conn:
            _insert(
                conn,
                "hotel_followups",
                {
                    "code": code,
                    "kind": kind,
                    "caller_name": caller_name,
                    "caller_phone": normalize_phone(caller_phone),
                    "summary": summary,
                },
            )
        await self._notify_change()
        return code

    async def schedule_wakeup_call(
        self,
        *,
        room: str,
        guest_name: str,
        call_date: date,
        call_time: time,
    ) -> str:
        room_id = self._require_room(room)
        conn = self.connection
        self._require_future(call_date)
        code = shortuuid("WUC-")
        with conn:
            _insert(
                conn,
                "wakeup_calls",
                {
                    "code": code,
                    "room_id": room_id,
                    "guest_name": guest_name,
                    "date": call_date.isoformat(),
                    "time": call_time.isoformat(),
                },
            )
        await self._notify_change()
        return code

    async def book_tour(
        self,
        *,
        tour_id: str,
        guest_name: str,
        guest_phone: str,
        on_date: date,
        party_size: int,
    ) -> tuple[str, Tour, int]:
        tour = TOURS.get(tour_id)
        if tour is None:
            raise NotFound(f"no such tour: {tour_id} - options: {', '.join(TOURS)}")
        self._require_future(on_date)
        if party_size > tour.max_party:
            raise Unavailable(f"{tour.name} takes at most {tour.max_party} guests")
        total = tour.flat_price or (tour.price_per_person or 0) * party_size
        code = shortuuid("TUR-")
        with self.connection as conn:
            _insert(
                conn,
                "tour_bookings",
                {
                    "code": code,
                    "tour_id": tour_id,
                    "guest_name": guest_name,
                    "guest_phone": normalize_phone(guest_phone),
                    "date": on_date.isoformat(),
                    "party_size": party_size,
                    "total": total,
                },
            )
        await self._notify_change()
        return code, tour, total

    async def book_spa_appointment(
        self,
        *,
        service_id: str,
        guest_name: str,
        guest_phone: str,
        on_date: date,
        at_time: time,
        party_size: int,
    ) -> tuple[str, SpaService, int]:
        service = SPA_SERVICES.get(service_id)
        if service is None:
            raise NotFound(
                f"no such spa service: {service_id} - options: {', '.join(SPA_SERVICES)}"
            )
        self._require_future(on_date)
        if party_size > service.max_party:
            raise Unavailable(f"{service.name} takes at most {service.max_party} guests")
        total = service.price * party_size
        code = shortuuid("SPA-")
        with self.connection as conn:
            _insert(
                conn,
                "spa_bookings",
                {
                    "code": code,
                    "service_id": service_id,
                    "guest_name": guest_name,
                    "guest_phone": normalize_phone(guest_phone),
                    "date": on_date.isoformat(),
                    "time": at_time.isoformat(),
                    "party_size": party_size,
                    "total": total,
                },
            )
        await self._notify_change()
        return code, service, total

    async def book_business_center(
        self,
        *,
        service_id: str,
        guest_name: str,
        guest_phone: str,
        on_date: date,
        at_time: time,
        duration_hours: int,
    ) -> tuple[str, BusinessCenterService, int]:
        """Book a business-centre service, which is available 24 hours a day."""
        service = BUSINESS_CENTER_SERVICES.get(service_id)
        if service is None:
            raise NotFound(
                f"no such service: {service_id} - options: {', '.join(BUSINESS_CENTER_SERVICES)}"
            )
        self._require_future(on_date)
        if duration_hours > service.max_hours:
            raise Unavailable(f"{service.name} is booked for at most {service.max_hours} hours")
        total = service.flat_price or (service.price_per_hour or 0) * duration_hours
        code = shortuuid("BIZ-")
        with self.connection as conn:
            _insert(
                conn,
                "business_center_bookings",
                {
                    "code": code,
                    "service_id": service_id,
                    "guest_name": guest_name,
                    "guest_phone": normalize_phone(guest_phone),
                    "date": on_date.isoformat(),
                    "time": at_time.isoformat(),
                    "duration_hours": duration_hours,
                    "total": total,
                },
            )
        await self._notify_change()
        return code, service, total

    async def order_flowers(
        self,
        *,
        arrangement_id: str,
        guest_name: str,
        guest_phone: str,
        deliver_to: str,
        on_date: date,
        card_message: str,
    ) -> tuple[str, FloralArrangement, int]:
        arrangement = FLORIST_ARRANGEMENTS.get(arrangement_id)
        if arrangement is None:
            raise NotFound(
                f"no such arrangement: {arrangement_id} - options: "
                f"{', '.join(FLORIST_ARRANGEMENTS)}"
            )
        self._require_future(on_date)
        total = arrangement.price
        code = shortuuid("FLR-")
        with self.connection as conn:
            _insert(
                conn,
                "florist_orders",
                {
                    "code": code,
                    "arrangement_id": arrangement_id,
                    "guest_name": guest_name,
                    "guest_phone": normalize_phone(guest_phone),
                    "deliver_to": deliver_to,
                    "date": on_date.isoformat(),
                    "message": card_message,
                    "total": total,
                },
            )
        await self._notify_change()
        return code, arrangement, total

    async def send_email(self, *, recipient: str, kind: str) -> str:
        """Stub email send: records that a document of `kind` was sent to `recipient`
        and returns a reference. No real mail goes out; the row is the gradable signal
        that the agent actually sent rather than just claiming to."""
        if kind not in get_args(EmailKind):
            raise NotFound(
                f"unknown email kind: {kind} - options: {', '.join(get_args(EmailKind))}"
            )
        code = shortuuid("EML-")
        with self.connection as conn:
            _insert(
                conn,
                "emails_sent",
                {
                    "code": code,
                    "recipient": normalize_email(recipient),
                    "kind": kind,
                },
            )
        await self._notify_change()
        return code

    async def transfer_call(self, *, destination: str, summary: str) -> str:
        """Stub call transfer: records that the caller was transferred to a hotel
        department with a one-line summary, and returns a reference."""
        if destination not in get_args(TransferDestination):
            raise NotFound(
                f"unknown destination: {destination} - options: {', '.join(get_args(TransferDestination))}"
            )
        code = shortuuid("XFR-")
        with self.connection as conn:
            _insert(
                conn,
                "transfer_calls",
                {
                    "code": code,
                    "destination": destination,
                    "summary": summary,
                },
            )
        await self._notify_change()
        return code

    async def request_flight_reconfirmation(
        self,
        *,
        room: str,
        airline: str,
        flight_number: str,
        flight_date: date,
        booking_reference: str,
        seat_check: bool,
    ) -> str:
        room_id = self._require_room(room)
        code = shortuuid("FLT-")
        with self.connection as conn:
            _insert(
                conn,
                "flight_reconfirmations",
                {
                    "code": code,
                    "room_id": room_id,
                    "airline": airline.strip().title(),
                    # spoken codes arrive with unpredictable spaces/dashes
                    "flight_number": "".join(c for c in flight_number if c.isalnum()).upper(),
                    "flight_date": flight_date.isoformat(),
                    "booking_reference": "".join(
                        c for c in booking_reference if c.isalnum()
                    ).upper(),
                    "seat_check": int(seat_check),
                },
            )
        await self._notify_change()
        return code

    async def book_airport_car(
        self,
        *,
        room: str,
        pickup_date: date,
        pickup_time: time,
        passengers: int,
    ) -> str:
        room_id = self._require_room(room)
        self._require_future(pickup_date)
        code = shortuuid("CAR-")
        with self.connection as conn:
            _insert(
                conn,
                "airport_cars",
                {
                    "code": code,
                    "room_id": room_id,
                    "pickup_date": pickup_date.isoformat(),
                    "pickup_time": pickup_time.isoformat(),
                    "passengers": passengers,
                },
            )
        await self._notify_change()
        return code

    async def dispatch_emergency(self, *, room: str, kind: str, situation: str) -> str:
        # A bad kind is an invalid argument, not a missing entity - keep it distinct
        # from the room's NotFound so the tool can't misreport it as a bad room number.
        if kind not in get_args(EmergencyKind):
            raise ValueError(
                f"unknown emergency kind: {kind} - options: {', '.join(get_args(EmergencyKind))}"
            )
        room_id = self._require_room(room)
        code = shortuuid("EMG-")
        with self.connection as conn:
            _insert(
                conn,
                "emergency_dispatches",
                {"code": code, "room_id": room_id, "kind": kind, "situation": situation},
            )
        await self._notify_change()
        return code

    async def room_conflict(self, *, booking_code: str) -> tuple[date, date] | None:
        """The overlap window if another confirmed booking holds this booking's room."""
        row = self.connection.execute(_SQL_ROOM_CONFLICT, {"code": booking_code}).fetchone()
        if not row:
            return None
        return date.fromisoformat(row[0]), date.fromisoformat(row[1])

    async def resolve_room_conflict(self, *, booking_code: str) -> ConflictResolution:
        """The house re-accommodation procedure, in fixed order: try to move the
        booking to a free room of the same or higher category for the whole
        (remaining) stay - an upgrade is free - and only when nothing in the
        house fits, arrange a walk: tonight at the partner hotel on us, back in
        the original room from the next day."""
        conn = self.connection
        booking = conn.execute(
            "SELECT room_id, check_in, check_out, guests FROM hotel_bookings"
            " WHERE code = :code AND status = 'confirmed'",
            {"code": booking_code},
        ).fetchone()
        if not booking:
            raise NotFound(f"booking not found: {booking_code}")
        if await self.room_conflict(booking_code=booking_code) is None:
            raise Unavailable("no room conflict on this booking - nothing to resolve")
        room_id, check_in, check_out, guests = booking
        original = conn.execute(
            "SELECT smoking, nightly_rate FROM hotel_rooms WHERE id = :id", {"id": room_id}
        ).fetchone()
        assert original is not None
        start = max(date.fromisoformat(check_in), self.today)

        candidate = conn.execute(
            _SQL_FREE_BETTER_ROOM,
            {
                "exclude_room": room_id,
                "exclude_code": booking_code,
                "guests": guests,
                "smoking": original[0],
                "min_rate": original[1],
                "check_in": start.isoformat(),
                "check_out": check_out,
            },
        ).fetchone()

        if candidate:
            new_room, new_type, new_view, new_rate = candidate
            with conn:
                # the rate on the booking doesn't change - a forced move is never
                # the guest's cost, so an upgrade rides at the original total
                _update(conn, "hotel_bookings", {"room_id": new_room}, {"code": booking_code})
            await self._notify_change()
            return ConflictResolution(
                moved_to=new_room,
                moved_to_type=new_type,
                moved_to_view=new_view,
                upgraded=new_rate > original[1],
            )

        return_date = start + timedelta(days=1)
        with conn:
            _insert(
                conn,
                "walk_arrangements",
                {
                    "code": shortuuid("WLK-"),
                    "booking_code": booking_code,
                    "partner_hotel": WALK_PARTNER_HOTEL,
                    "return_date": return_date.isoformat(),
                },
            )
        await self._notify_change()
        return ConflictResolution(walk_partner=WALK_PARTNER_HOTEL, walk_return_date=return_date)

    def _require_room(self, room: str) -> str:
        """Normalize a spoken room number ("304") to its id and require it exists."""
        room_id = room.strip().upper()
        if not room_id.startswith("RM_"):
            room_id = f"RM_{room_id}"
        if not self.connection.execute(
            "SELECT 1 FROM hotel_rooms WHERE id = :id", {"id": room_id}
        ).fetchone():
            raise NotFound(f"no such room: {room}")
        return room_id

    async def take_guest_message(
        self,
        *,
        recipient: str,
        caller_name: str,
        caller_phone: str,
        message: str,
    ) -> str:
        """Record a message addressed to a (possibly) in-house guest. Whether the
        recipient actually has a stay here is resolved internally and never
        returned, so the agent taking the message cannot leak guest presence."""
        code = shortuuid("MSG-")
        conn = self.connection
        with conn:
            in_house = conn.execute(
                "SELECT first_name || ' ' || last_name FROM hotel_bookings"
                " WHERE status = 'confirmed'"
                " AND LOWER(first_name || ' ' || last_name) = LOWER(TRIM(:name))"
                " AND check_in <= :today AND check_out > :today",
                {"name": recipient, "today": self.today.isoformat()},
            ).fetchone()
            _insert(
                conn,
                "guest_messages",
                {
                    "code": code,
                    # matched messages take the registered guest's casing so the
                    # stored name doesn't depend on how the caller's was heard
                    "recipient": in_house[0] if in_house else recipient,
                    "caller_name": caller_name,
                    "caller_phone": normalize_phone(caller_phone),
                    "message": message,
                    "status": "delivered" if in_house else "undeliverable",
                },
            )
        await self._notify_change()
        return code

    async def peek_stay_total(
        self,
        *,
        room_type: str,
        smoking: bool,
        guests: int,
        check_in: date,
        check_out: date,
        view: str | None,
        extras: Sequence[str],
    ) -> int | None:
        """The exact total (with tax) for the room book_room would pick right now -
        so the agent can quote the real number in the read-back instead of doing
        per-night arithmetic itself (and forgetting tax)."""
        row = self.connection.execute(
            _SQL_FREE_ROOM,
            {
                "room_type": room_type,
                "smoking": int(smoking),
                "guests": guests,
                "check_in": check_in.isoformat(),
                "check_out": check_out.isoformat(),
                "exclude": None,
                "view": view,
                "prefer": None,
            },
        ).fetchone()
        if not row:
            return None
        nights = (check_out - check_in).days
        _, _, total, _ = compute_invoice(nightly_rate=row[1], nights=nights, extras=list(extras))
        return total

    async def record_group_inquiry(
        self,
        *,
        company: str,
        contact_name: str,
        contact_phone: str,
        party_size: int,
        share_type: GroupShareType,
        check_in: date,
        nights: int,
    ) -> str:
        code = shortuuid("GRP-")
        conn = self.connection
        with conn:
            _insert(
                conn,
                "group_inquiries",
                {
                    "code": code,
                    "company": company,
                    "contact_name": contact_name,
                    # digits only: a spoken callback number transcribes with
                    # unpredictable punctuation, and nothing dials it back out
                    "contact_phone": normalize_phone(contact_phone),
                    "party_size": party_size,
                    "share_type": share_type,
                    "check_in": check_in.isoformat(),
                    "nights": nights,
                },
            )
        await self._notify_change()
        return code

    async def file_dispute(
        self,
        *,
        booking_code: str,
        line_item: str,
        amount_cents: int,
        category: DisputeCategory,
        caller_note: str,
        outcome: str,
        refund_amount: int,
    ) -> str:
        case_number = shortuuid("DSP-")
        conn = self.connection
        with conn:
            _insert(
                conn,
                "hotel_disputes",
                {
                    "case_number": case_number,
                    "booking_code": booking_code,
                    "line_item": line_item,
                    "amount": amount_cents,
                    "category": category,
                    "caller_note": caller_note,
                    "outcome": outcome,
                    "refund_amount": refund_amount,
                },
            )
            if refund_amount > 0:
                conn.execute(
                    "UPDATE hotel_invoices SET total = total - :refund WHERE booking_code = :code",
                    {"refund": refund_amount, "code": booking_code},
                )
        await self._notify_change()
        return case_number
