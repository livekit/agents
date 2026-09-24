# Hotel Guest Support Example

A guest-support agent for a boutique hotel: requests, complaints, and emergencies
from guests who are staying, arriving, or have just left. It's one of three focused
hotel agents, along with [reservations](../hotel_reservations/) and
[amenities](../hotel_amenities/), split out of the all-in-one
[hotel receptionist](../hotel_receptionist/).

## What it handles

- Emergencies (medical, fire, security): staff are dispatched to the room first
- Messages for guests, without ever confirming who is staying at the hotel
- Wake-up calls and do-not-disturb holds
- Housekeeping and maintenance requests, lost and found, early checkout
- Double-booked rooms: a free move or upgrade, or a walk to the partner hotel
- Transfers to the restaurant, duty manager, or housekeeping

## Running

```bash
uv run examples/hotel_guest_support/agent.py console
```

The hotel data is an in-memory SQLite database seeded from `seed.py` on every call.
To try the double-booking flow, use last name `Whelan` with code `HTL-TW55`.

## Simulations

`scenarios.yaml` holds 29 scenarios. The 18 with `userdata.expected_state` are also
graded on the final database state (`benchmark.py`), on top of the conversation.

```bash
lk agent simulate examples/hotel_guest_support/agent.py \
  --scenarios examples/hotel_guest_support/scenarios.yaml
```

## Files

```
agent.py           GuestSupportAgent, its tools, and simulation grading
instructions.py    Persona and routing instructions
verify_booking.py  Caller verification task
hotel_db.py        Schema and queries
seed.py            Seed data
benchmark.py       Final-state diff used to grade simulations
policies/          Long-tail policy text served by lookup_policy
```
