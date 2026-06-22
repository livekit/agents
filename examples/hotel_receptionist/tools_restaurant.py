from __future__ import annotations

import logging
import os
import sys
from datetime import date, time
from typing import Annotated

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from book_restaurant import BookRestaurantTask
from common import Userdata, _speak_code
from context import speech_only
from hotel_db import (
    MAX_PARTY_SIZE,
    Unavailable,
    speak_time,
)
from pydantic import Field

from livekit.agents import RunContext, ToolError, function_tool

logger = logging.getLogger("hotel-receptionist")


class RestaurantToolsMixin:
    @function_tool
    async def check_restaurant_availability(
        self,
        ctx: RunContext[Userdata],
        on_date: date,
        party_size: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)],
    ) -> str:
        """Check restaurant time slots for a date. Read-only browsing - to actually book a table, call start_restaurant_booking.

        Args:
            on_date: The date to check, in ISO YYYY-MM-DD format (e.g. "2026-01-20").
            party_size: Number of guests (must be >= 1; ask the caller if not specified).
        """
        slots = await ctx.userdata.db.list_restaurant_availability(
            on_date=on_date, party_size=party_size
        )
        open_slots = [s for s in slots if s.available_table_ids]
        if not open_slots:
            return f"fully booked on {on_date.strftime('%A, %B %-d')}"
        return ", ".join(speak_time(s.time) for s in open_slots)

    @function_tool
    async def start_restaurant_booking(self, ctx: RunContext[Userdata]) -> str | None:
        """Start the restaurant-reservation flow. Call it the moment the caller wants a table - the flow collects date, party size, time, name, and phone itself. Its return is the FINAL result of the reservation: relay it and move on - nothing further to confirm or call afterwards."""
        reservation = await BookRestaurantTask(
            db=ctx.userdata.db, chat_ctx=speech_only(self.chat_ctx)
        )
        ctx.userdata.booked_restaurant_codes.append(reservation.code)
        return (
            f"You're set for {speak_time(reservation.time)} on "
            f"{reservation.date.strftime('%A, %B %-d')} for "
            f"{reservation.party_size} guest{'s' if reservation.party_size != 1 else ''}. "
            f"Confirmation code: {_speak_code(reservation.code)}. "
            "| reservation complete - relay this to the caller; no further tool call is needed."
        )

    @function_tool
    async def lookup_restaurant_reservation(
        self,
        ctx: RunContext[Userdata],
        last_name: str,
        confirmation_code: str,
    ) -> str:
        """Read-only lookup of a confirmed restaurant reservation. Use this when the caller wants
        to check or recall their reservation details (date, time, party size, notes) without
        changing or cancelling it.

        Args:
            last_name: caller's last name.
            confirmation_code: confirmation code like 'RES-X9Y2'.
        """
        code = confirmation_code.replace(" ", "").upper()
        reservation = await ctx.userdata.db.find_restaurant_reservation(
            last_name=last_name, confirmation_code=code
        )
        if not reservation or reservation.status != "confirmed":
            raise ToolError("Couldn't find a matching confirmed reservation.")
        notes_part = f", note: {reservation.notes}" if reservation.notes else ""
        return (
            f"Reservation for {reservation.first_name} {reservation.last_name}, "
            f"{speak_time(reservation.time)} on {reservation.date.strftime('%A, %B %-d')}, "
            f"party of {reservation.party_size}{notes_part}."
        )

    @function_tool
    async def cancel_restaurant_reservation(
        self,
        ctx: RunContext[Userdata],
        last_name: str,
        confirmation_code: str,
    ) -> str:
        """Cancel a restaurant reservation. Restaurants verify with last name + confirmation code (no card, no email — the code is what we print when the table is booked).

        Args:
            last_name: caller's last name.
            confirmation_code: confirmation code like 'RES-X9Y2'.
        """
        code = confirmation_code.replace(" ", "").upper()
        reservation = await ctx.userdata.db.find_restaurant_reservation(
            last_name=last_name, confirmation_code=code
        )
        if not reservation or reservation.status != "confirmed":
            raise ToolError("Couldn't find a matching confirmed reservation.")
        await ctx.userdata.db.cancel_restaurant_reservation(reservation.code)
        ctx.userdata.cancelled_codes.append(reservation.code)
        return (
            f"Reservation for {speak_time(reservation.time)} on "
            f"{reservation.date.strftime('%A, %B %-d')} cancelled."
        )

    @function_tool
    async def modify_restaurant_reservation(
        self,
        ctx: RunContext[Userdata],
        last_name: str,
        confirmation_code: str,
        new_date: date,
        new_time: str,
        new_party_size: Annotated[int, Field(ge=1, le=MAX_PARTY_SIZE)] | None = None,
    ) -> str:
        """Move an existing confirmed restaurant reservation to a new date/time (and
        optionally a new party size), keeping the same confirmation code. Restaurants
        verify with last name + confirmation code (no card, no email). Read the new
        details back to the caller before calling this.

        Args:
            last_name: caller's last name.
            confirmation_code: confirmation code like 'RES-X9Y2'.
            new_date: the new date, in ISO YYYY-MM-DD format (e.g. "2026-01-20").
            new_time: the new time, in 24-hour HH:MM format (e.g. "18:00").
            new_party_size: new number of guests; omit to keep the current party size.
        """
        code = confirmation_code.replace(" ", "").upper()
        reservation = await ctx.userdata.db.find_restaurant_reservation(
            last_name=last_name, confirmation_code=code
        )
        if not reservation or reservation.status != "confirmed":
            raise ToolError("Couldn't find a matching confirmed reservation.")
        try:
            at_time = time.fromisoformat(new_time)
        except ValueError:
            raise ToolError("Please give the new time as 24-hour HH:MM, e.g. 18:00.") from None
        try:
            updated = await ctx.userdata.db.modify_restaurant_reservation(
                code=reservation.code,
                on_date=new_date,
                at_time=at_time,
                party_size=new_party_size,
            )
        except Unavailable:
            raise ToolError(
                f"No table for a party of {new_party_size or reservation.party_size} "
                f"at {speak_time(at_time)} on {new_date.strftime('%A, %B %-d')}."
            ) from None
        ctx.userdata.booked_restaurant_codes.append(updated.code)
        return (
            f"Done - your reservation is now {speak_time(updated.time)} on "
            f"{updated.date.strftime('%A, %B %-d')} for "
            f"{updated.party_size} guest{'s' if updated.party_size != 1 else ''}, "
            f"under confirmation code {_speak_code(updated.code)}."
        )
