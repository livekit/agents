# Hotel Receptionist Example

A boutique-hotel receptionist agent. Handles room bookings, restaurant
reservations, cancellations, invoices, charge disputes, and FAQs about
the hotel — backed by a per-call, in-memory SQLite database that streams
its state to the playground in real time.

## Running

```bash
python -m examples.hotel_receptionist.agent console
# or, with the LiveKit playground:
python -m examples.hotel_receptionist.agent dev
```

Each call starts with a fresh database built from `fake_data/seed.py`; the
database is discarded when the call ends. Running the seed script separately
is not required. For manual inspection, you can run
`python -m examples.hotel_receptionist.fake_data.seed [path/to/hotel.db]` to
write a standalone SQLite file. The agent does not read that file.

## Architecture

```
agent.py           — runtime assembly and entrypoint
evaluation.py      — simulation grading and run reports
hotel_agent.py     — shared agent state contract
tools_*.py         — room, restaurant, and service tools
book_*.py          — focused booking tasks
hotel.py           — domain types, catalogs, pricing, and formatting
hotel_db.py        — HotelDB operations
hotel_schema.py    — SQLite schema and row conversion
instructions.py    — prompt instructions and routing rules
ui_view.py         — in-memory SQLite mirror for the playground
fake_data/seed.py  — seed builder and optional database writer
```

The LLM never owns money values. `book_room` computes the total
server-side from `nightly_rate × nights + extras_total(...) + tax`.
`file_dispute` reads the disputed amount from the stored invoice line
item by label and clamps any refund to its amount.

External side effects are simulated. For example, sending an email or
transferring a call records the action in SQLite; it does not contact a real
email or telephony service. This keeps the example focused on agent behavior,
tool selection, and state transitions.

## In-memory DB → playground

`ui_view.py` attaches an `apsw.Session` to the DB connection and
captures binary changesets after every commit. A subscribe handshake
gives the playground browser a base snapshot via byte stream, then
each subsequent write is sent as a `sqlite_diff` RPC carrying the
binary changeset. The demo frontend's matching `useSqliteMirror` hook
applies those changesets in the browser.
