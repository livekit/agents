# Hotel Amenities Example

An amenities agent for a boutique hotel: the restaurant, the spa, the business
centre, the florist, sightseeing tours, and the hotel car. It's one of three focused
hotel agents, along with [reservations](../hotel_reservations/) and
[guest support](../hotel_guest_support/), split out of the all-in-one
[hotel receptionist](../hotel_receptionist/).

## What it handles

- Restaurant tables: availability, booking (`BookRestaurantTask`), changes, and cancellations
- Large-party and private dining, transferred to the restaurant
- Spa and health-club appointments, business-centre bookings, and flower orders
- Sightseeing tours, flight reconfirmation, and the hotel car to the airport
- Event and wedding inquiries, recorded for the sales team

## Running

```bash
uv run examples/hotel_amenities/agent.py console
```

The hotel data is an in-memory SQLite database seeded from `seed.py` on every call.
To try the reservation flows, use last name `Bennett` with code `RES-JK90`.

## Simulations

`scenarios.yaml` holds 19 scenarios. The 13 with `userdata.expected_state` are also
graded on the final database state (`benchmark.py`), on top of the conversation.

```bash
lk agent simulate examples/hotel_amenities/agent.py \
  --scenarios examples/hotel_amenities/scenarios.yaml
```

## Files

```
agent.py            AmenitiesAgent, its tools, and simulation grading
instructions.py     Persona and routing instructions
book_restaurant.py  Restaurant-reservation task
hotel_db.py         Schema, catalogs, and queries
seed.py             Seed data
benchmark.py        Final-state diff used to grade simulations
policies/           Long-tail policy text served by lookup_policy
```
