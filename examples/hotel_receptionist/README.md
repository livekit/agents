# Hotel Receptionist Example

A boutique-hotel receptionist agent. Handles room bookings, restaurant
reservations, cancellations, invoices, charge disputes, and FAQs about
the hotel — backed by a live SQLite database that streams its state to
the playground in real time.

## Running

```bash
python examples/hotel_receptionist/fake_data/seed.py
python examples/hotel_receptionist/agent.py console
# or, with the LiveKit playground:
python examples/hotel_receptionist/agent.py dev
```

`fake_data/seed.py` prints sample confirmation codes you can use to try
the cancellation/invoice/dispute flows immediately, e.g. `Cancel: last
name 'Smith', code 'HTL-AB12'`.

## Architecture

```
agent.py           — HotelReceptionistAgent + tool mixins
tools_*.py         — Tool mixins: rooms, restaurant, services
book_*.py          — AgentTask subclasses for booking flows
hotel_db.py        — HotelDB (apsw) + schema + views + pricing + dispute policy
instructions.py    — Prompt instructions and routing rules
ui_view.py         — SQLite changeset streamer for the playground
fake_data/seed.py  — manual seed script (writes fake_data/hotel.db)
```

The LLM never owns money values. `book_room` computes the total
server-side from `rooms.nightly_rate × nights + PRICING.extras(...) +
tax`. `file_dispute` reads the disputed amount from the stored invoice
line item by label and clamps any refund to ≤ amount.

## Live DB → playground

`sqlite_diff.py` attaches an `apsw.Session` to the DB connection and
captures binary changesets after every commit. A subscribe handshake
gives each playground browser a base snapshot via byte stream, then
each subsequent write fans out as a `sqlite_diff` RPC carrying the
binary changeset. See the matching `useSqliteMirror` hook in the
jukebox repo.
