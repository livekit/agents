# Hotel Reservations Example

A reservations agent for a boutique hotel: room bookings, changes, cancellations,
and the payments and billing on them. It's one of three focused hotel agents, along
with [guest support](../hotel_guest_support/) and [amenities](../hotel_amenities/),
split out of the all-in-one [hotel receptionist](../hotel_receptionist/).

## What it handles

- Checking availability and booking a room (`BookRoomTask`, including card capture)
- Changing dates, room, view, extras, or party size on a booking (`ModifyBookingTask`)
- Cancelling, reinstating a cancelled booking, and waitlisting sold-out dates
- Group room blocks (15+ guests), late arrivals, and returning-guest preferences
- Replacing the card on file, invoice lookups, charge disputes, and re-sending a folio

Existing bookings are verified by last name plus confirmation code, or last name plus
the card's last four (`VerifyBookingTask`).

## Running

```bash
uv run examples/hotel_reservations/agent.py console
```

The hotel data is an in-memory SQLite database seeded from `seed.py` on every call.
To try the existing-booking flows, use last name `Smith` with code `HTL-AB12`.

## Simulations

`scenarios.yaml` holds 52 scenarios. The 26 with `userdata.expected_state` are also
graded on the final database state (`benchmark.py`), on top of the conversation.

```bash
lk agent simulate examples/hotel_reservations/agent.py \
  --scenarios examples/hotel_reservations/scenarios.yaml
```

## Files

```
agent.py           ReservationsAgent, its tools, and simulation grading
instructions.py    Persona and routing instructions
book_room.py       Room-booking task
modify_booking.py  Booking-modification task
verify_booking.py  Caller verification task
get_card.py        Card-capture task
hotel_db.py        Schema, pricing, dispute policy, and queries
seed.py            Seed data
benchmark.py       Final-state diff used to grade simulations
policies/          Long-tail policy text served by lookup_policy
```
