# Pearl Dental

A dental practice's appointment scheduling as a fast brain and a slow brain, on the local
stack of [`../README.md`](../README.md): the openings, and a booking with the patient's
email, by text and on the phone.

| File                | Agent              | What                                                                                                  |
| ------------------- | ------------------ | ----------------------------------------------------------------------------------------------------- |
| `frontend_agent.py` | `dentist`         | Talks to the patient, on the phone and by text (the `frontend` endpoint). Its one tool is the back office. |
| `backend_agent.py`  | lane `backend` | The back office, reached over A2A: one agent with two tools on the calendar.                        |
| `calendar_api.py`   |                    | The calendar: a fake one seeded with a known week, or cal.com when `CAL_API_KEY` is set.              |
| `main.py`           |                   | The one entrypoint: the server, the two handlers added to it.                         |
| `scenarios/*.yaml`  |                    | Simulation scenarios: `booking`, `availability`.                                                      |

## What the pattern is made of

- **A delegate as a tool.** `Agent(tools=[A2ADelegate(BACKEND)])` on the frontend, for the
  call and the text door alike: one tool, `ask`, described from the back office's card. The
  frontend relays what comes back and never guesses.
- **A tool that asks for what it is missing.** `schedule_appointment` takes the slot and the
  email. Without the email it returns `InputRequired` with the question; the turn ends there,
  the frontend relays it, and the patient's answer is written as the call's output on the next
  turn, so the model calls the tool again with the email. With everything, and before writing,
  `requires_approval=read_back` reads the appointment back as written; the booking runs only
  when the model repeats the same call after the patient answered, so it cannot skip the ask.
- **The calendar as the backend's `deps`**, the same object for every turn of every
  conversation, and the system of record for what is booked. Its clock is read into the
  instructions on every turn, after the static part, so the model knows the date.
- **The lane's state** keeps the patient's email, so a second booking does not ask again.

The same pattern with guided flows as tasks awaited in text is [`../hotel-receptionist/`](../hotel-receptionist/README.md).

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
FRONTDESK_NOW=2026-06-12T09:00 uv run python main.py dev
```

The console (`../../console`) lists this example and picks it once it is up.

`FRONTDESK_NOW` pins the practice's clock, so the seeded openings and the scenarios' absolute
dates stay where they are; without it the practice runs on the wall clock, with the same
openings as days from today. `CAL_API_KEY` swaps the fake calendar for cal.com.

## Scenarios

```sh
lk agent simulate text --scenarios scenarios/booking.yaml --agent-name dentist
```

`booking` covers the paths that book: a patient hesitant about the email, one with a date in
mind, a change of mind before the read-back, the earliest opening, and a listing after a
booking that must no longer show the slot taken. `availability` covers the pushes for times
the practice does not have: a morning that is not open, a demand with no acceptable alternative,
a weekend, minutes around a listed time, a month with nothing. Every scenario that books takes
a different slot, since the calendar is shared across a run; restart the backend between runs.
