from __future__ import annotations

import base64
import datetime
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import aiohttp

from livekit.agents.utils import http_context

TIMEZONE = ZoneInfo("UTC")
SLOT_MINUTES = 30
# FRONTDESK_NOW pins the clock, so the scenarios' absolute dates stay where they are
NOW = (
    datetime.datetime.fromisoformat(os.environ["FRONTDESK_NOW"]).replace(tzinfo=TIMEZONE)
    if os.environ.get("FRONTDESK_NOW")
    else None
)


class SlotUnavailable(Exception):
    """The slot was taken between the listing and the booking."""


@dataclass(frozen=True)
class Slot:
    start: datetime.datetime

    @property
    def id(self) -> str:
        """A short id the model can repeat without mangling a datetime."""
        digest = hashlib.blake2s(self.start.isoformat().encode(), digest_size=5).digest()
        return "ST_" + base64.b32encode(digest).decode().rstrip("=").lower()


@dataclass
class Appointment:
    slot: Slot
    email: str


class Calendar(Protocol):
    appointments: list[Appointment]

    def now(self) -> datetime.datetime: ...
    async def available(self, start: datetime.datetime, end: datetime.datetime) -> list[Slot]: ...
    async def schedule(self, slot: Slot, email: str) -> None: ...


@dataclass
class FakeCalendar:
    slots: list[Slot]
    appointments: list[Appointment] = field(default_factory=list)

    @classmethod
    def seeded(cls) -> FakeCalendar:
        """Openings over the next five weeks, as days from today and an hour: a known week the
        scenarios are written against, pinned with FRONTDESK_NOW=2026-06-12T09:00, a Friday."""
        today = datetime.datetime.combine(cls.now().date(), datetime.time(), tzinfo=TIMEZONE)
        openings = [
            (0, 15.5),
            (3, 14), (3, 16.5),
            (4, 9), (4, 15),
            (5, 11), (5, 14.5),
            (6, 11), (6, 14.5),
            (7, 10),
            (10, 13),
            (12, 15),
            (34, 9.5), (34, 13),
            (35, 10),
        ]  # fmt: skip
        return cls([Slot(today + datetime.timedelta(days=d, hours=h)) for d, h in openings])

    @staticmethod
    def now() -> datetime.datetime:
        return NOW or datetime.datetime.now(TIMEZONE)

    async def available(self, start: datetime.datetime, end: datetime.datetime) -> list[Slot]:
        return [s for s in self.slots if start <= s.start < end]

    async def schedule(self, slot: Slot, email: str) -> None:
        if slot not in self.slots:
            raise SlotUnavailable
        self.slots.remove(slot)
        self.appointments.append(Appointment(slot, email))


CAL_COM = "https://api.cal.com/v2/"
EVENT_SLUG = "livekit-front-desk"


class CalComCalendar:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._event_id: int | None = None
        self.appointments: list[Appointment] = []
        try:
            self._http = http_context.http_session()
        except RuntimeError:
            self._http = aiohttp.ClientSession()

    def now(self) -> datetime.datetime:
        return datetime.datetime.now(TIMEZONE)

    def _headers(self, version: str) -> dict[str, str]:
        return {"cal-api-version": version, "Authorization": f"Bearer {self._api_key}"}

    async def _event(self) -> int:
        """The id of the front desk's event type, created the first time."""
        if self._event_id is not None:
            return self._event_id
        async with self._http.get(CAL_COM + "me/", headers=self._headers("2024-06-14")) as resp:
            resp.raise_for_status()
            username = (await resp.json())["data"]["username"]
        query = urlencode({"username": username})
        async with self._http.get(
            CAL_COM + f"event-types/?{query}", headers=self._headers("2024-06-14")
        ) as resp:
            resp.raise_for_status()
            found = next(
                (e for e in (await resp.json())["data"] if e.get("slug") == EVENT_SLUG), None
            )
        if found is None:
            async with self._http.post(
                CAL_COM + "event-types",
                headers=self._headers("2024-06-14"),
                json={
                    "lengthInMinutes": SLOT_MINUTES,
                    "title": "LiveKit Front-Desk",
                    "slug": EVENT_SLUG,
                },
            ) as resp:
                resp.raise_for_status()
                found = (await resp.json())["data"]
            logging.getLogger("cal.com").info("created the %s event type", EVENT_SLUG)
        self._event_id = int(found["id"])
        return self._event_id

    async def available(self, start: datetime.datetime, end: datetime.datetime) -> list[Slot]:
        query = urlencode(
            {
                "eventTypeId": await self._event(),
                "start": start.astimezone(datetime.timezone.utc).isoformat(),
                "end": end.astimezone(datetime.timezone.utc).isoformat(),
            }
        )
        async with self._http.get(
            CAL_COM + f"slots/?{query}", headers=self._headers("2024-09-04")
        ) as resp:
            resp.raise_for_status()
            days = (await resp.json())["data"]
        return [
            Slot(datetime.datetime.fromisoformat(s["start"]).astimezone(TIMEZONE))
            for slots in days.values()
            for s in slots
        ]

    async def schedule(self, slot: Slot, email: str) -> None:
        async with self._http.post(
            CAL_COM + "bookings",
            headers=self._headers("2024-08-13"),
            json={
                "start": slot.start.astimezone(datetime.timezone.utc).isoformat(),
                "attendee": {"name": email, "email": email, "timeZone": str(TIMEZONE)},
                "eventTypeId": await self._event(),
            },
        ) as resp:
            data = await resp.json()
            if "already has booking" in str(data.get("error", {}).get("message", "")):
                raise SlotUnavailable
            resp.raise_for_status()
        self.appointments.append(Appointment(slot, email))


def calendar() -> Calendar:
    if key := os.environ.get("CAL_API_KEY"):
        return CalComCalendar(key)
    return FakeCalendar.seeded()
