# The LiveKit Hotel

A hotel receptionist as a fast brain and a slow brain, on the local stack of
[`../README.md`](../README.md): room bookings, lookups and cancellations, by text and on the
phone.

| File                | Agent            | What                                                                                                       |
| ------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------- |
| `frontend_agent.py` | `hotel`         | Talks to the guest, on the phone and by text (the `frontend` endpoint). Knows nothing about the hotel; its one tool is the back office. |
| `backend_agent.py`  | lane `backend` | The back office, reached over A2A: one routing agent with four tools, and two guided flows as tasks.      |
| `hotel.py`          |                  | The hotel's own system: rooms, bookings and prices, in memory. Reached over HTTP for real.                |
| `main.py`           |                  | The one entrypoint: the server, the two handlers added to it.                                             |
| `scenarios/*.yaml`  |                  | Simulation scenarios, one file per flow: `room_booking`, `lookup_and_cancel`, `availability`.             |

## What the pattern is made of

- **A delegate as a tool.** `Agent(tools=[A2ADelegate(BACKEND)])` on the frontend, for the
  call and the text door alike: one tool, `ask`, described from the back office's card. The
  frontend relays what comes back and never guesses.
- **Tasks awaited in text** for the flows that take several turns. `BookRoom` collects the
  stay, the room, the guest and the card, reads the booking back and writes it on the guest's
  yes; `VerifyBooking` finds the booking from the last name and code, three attempts. A routing
  tool returns `await BookRoom(...)`: the task rides the lane's agent stack with its arguments
  and its state, each message is a turn of it, and its `Completed(text)` is the call's output.
  A tool's code after the await does not run in text, so the task completes with the text the
  routing agent reads.
- **The guest's yes before a write.** `confirm_booking` and `cancel_room_booking` are
  `requires_approval=` with a read-back function of their own parameters: the booking from
  the task's state, the cancellation with its refund. A call returns the read-back as written
  and does not run; the guest's answer is written as the call's output on the next turn, and
  the write runs only when the model repeats the same call after it. The model cannot approve
  for the guest: there is no flag to set.
- **The lane's state** keeps what outlives a flow: the verified booking and the last booking
  made, rendered into the routing agent's instructions on every turn.

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
HOTEL_TODAY=2026-11-02 uv run python main.py dev
```

The console (`../../console`) lists this example and picks it once it is up.

`HOTEL_TODAY` pins the hotel's clock, so the seeded bookings and the scenarios' absolute dates
stay where they are; without it the hotel runs on today's date.

The console (`../../console`) then talks to the hotel. Try: a room for two next week; the
booking under Smith, code HTL-AB12 (its dates, then a cancellation); what is free the 8th to
the 10th.

## Scenarios

`scenarios/` holds the simulation sets, one file per flow, for `lk agent simulate`:

```sh
lk agent simulate text --scenarios scenarios/room_booking.yaml --agent-name hotel
```

Each covers the path that carries traffic, its edges (a detail corrected during the
read-back, a change of mind, nothing available, a party too large), a failed lookup, and the
refusals that must hold: no cancellation and no details without the confirmation code, and
nothing about another guest. The simulated guest follows one script template and reveals
facts only when asked; the expectations describe the outcome and name the failing case.

A restarted worker answers slowly for its first half minute: the server keeps the killed
worker's route until its websocket times out.
