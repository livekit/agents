# Review: a slow booking

A diagnostic example for the `live` inbox, background work, interrupt and cancel ([`docs/delegation.md`](../../docs/delegation.md), "When the agent is busy" and "When work stops"). A travel desk delegates to a booking office whose `book_flight` tool reports nine steps of 5 s each, then takes the seat in a region a cancel does not cut. Every step logs the status of its own task row as the store has it.

| File                | Agent           | What                                                                                                   |
| ------------------- | --------------- | ------------------------------------------------------------------------------------------------------ |
| `frontend_agent.py` | `booking`       | The travel desk, on the phone and by text (the `frontend` endpoint). Its one tool is a `LoggingDelegate`. |
| `backend_agent.py`  | lane `backend`  | The booking office: the slow `book_flight`, the instant `change_flight`, the inbox policy from the env.  |
| `tasks.py`          |                 | `tasks/list` on a thread: every task row with its status and metadata.                                  |
| `cancel.py`         |                 | `tasks/cancel` on one task, reason `review`.                                                           |
| `review_log.py`     |                 | How items and task rows are printed.                                                                   |
| `main.py`           |                 | The one entrypoint: the server, the two handlers added to it.                                          |
| `scenarios/*.yaml`  |                 | `slow_booking`: book, then change the day while the booking runs.                                      |

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
uv run python main.py dev                                          # live, 9 x 5 s
REVIEW_INBOX=queue uv run python main.py dev                       # the booking office before live
```

`REVIEW_INBOX` is the booking office's policy for every channel (`live`, `queue`, `interrupt`, `batch`, `concurrent`, `reject`); `REVIEW_FRONT_INBOX` is the travel desk's text door inbox (`live` by default; any other makes a call take the lane per voice turn). The scripts take the worker's environment (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`):

```sh
uv run python tasks.py AT_…              # every row on the thread
uv run python cancel.py AT_… <task id>   # cancel one booking office task
```

## What the logs say

- `backend turn <id>: thread=AT_… task=<full id> policy=live lease_wait=0.1s batch=1 message=…`: a booking office turn as it starts. `lease_wait` is the time from the message's arrival to the start of its turn: how long it waited for the floor.
- `backend turn <id>: own rows: …`: its rows newest first, as loaded, with `[background=…]`, `[interrupted=…]`, `[mergedInto=…]`, `[reason=…]`, `[cancelRequested=…]` from the row's metadata.
- `backend turn <id>: book_flight step 4/9 day=Monday row=working`: one step, and the row's status as the store has it at that moment.
- `backend turn <id>: book_flight taking the seat`, `… booked Monday`, or `… stopped by a cancel`.
- `backend turn <id>: done in 0.9s output=… interrupted=… booked_day=…`: the end of a turn. Under `live` the first turn of a booking ends after the first step, with the task still `working`: the booking runs on in the background, and a second turn of the same task (no message, `batch=1 message=''`) answers it when the tool ends.
- `frontend turn <id>: … lease_wait=…`, `delegate ask …`, `call tool …`, `call item added …`: the travel desk, as in [`../review-input-required/`](../review-input-required/README.md).

## What to try, and what to look for

**(a) A correction while the booking runs, by text.** Start the default worker. In the console chat, `book me a flight on Monday`. After the first `book_flight step` line, type `no, make it Tuesday`.

- The travel desk answers the first message at once ("on it"): its `ask` tool reported the booking office's first progress, went on in the background, and the turn ended with the task `working`. The console shows the task as working in the background.
- The correction runs at once (`frontend turn …: lease_wait=0.1s`), beside the booking: a second `delegate ask call_…: request='…Tuesday…'`, and a second `backend turn` with `lease_wait=0.xs` while the steps of the first still log. A delegation does not interrupt the expert (`livekit/interrupt: false`).
- `backend change_flight …: from=None to='Tuesday'` and later `book_flight booked Monday`: the expert's tools decide what a second request about work in flight means; this example's tools do not check, so Monday wins. A real `book_flight` would declare `on_duplicate=` or check the trip state.
- When the booking ends, the first chat task gets its answer: the console shows a second reply on the first task, and `tasks.py` shows the travel desk's first row `completed` with `[background=False]`.
- With `REVIEW_INBOX=queue`, the same: the second `backend turn` waits for the whole tool (`lease_wait=4x.xs`), as before this design.

**(b) A message that interrupts.** By text, ask something that makes the travel desk talk (`tell me about your baggage rules in detail`), and send `stop` while it streams.

- The first task ends `canceled` with `[interrupted=True]`: the reply stopped at its next boundary. The `stop` message runs next.
- Do the same while a booking runs in the background (a): the booking's steps keep logging. The interrupted travel desk task stays `working` while its background work runs, and ends when the work does.

**(c) `tasks/cancel` during a booking.** By text, `book me a flight on Monday`. Copy `thread=` and `task=` from the `backend turn` line, and while the steps log run `uv run python cancel.py AT_… <task>`.

- The next step line shows `row=working` with `cancelRequested` in `tasks.py`; then `book_flight stopped by a cancel, booked_day=None`: the tool stopped at its next wait point.
- `cancel.py` prints `TASK_STATE_CANCELED` once the tool stopped, with `{'…/reason': 'review'}`; no result is written after that.
- Run the cancel while the log says `taking the seat`: the seat is taken first (`booked_day='Monday'` in the `stopped by a cancel` line), because that region is `ctx.uninterruptible()`; then the task ends `canceled`.
- A second `cancel.py` on the same task answers an error: the task already ended.

**(d) On a call.** Say `book me a flight on Monday` and listen.

- `call tool started call_…: ask(…)`, the acknowledgment spoken at once, then each `booking step i/9` as `call tool updated` lines and spoken progress.
- Talk over the progress (`wait, stop`): the speech stops, `call tool ended` does not come early, and the booking office's steps keep logging: a barge-in stops the voice, not the work. Only `cancel.py` (or a tool the developer gives the model) cancels.
- Send a text in the console chat during the call: the call answers it (`call item added: user:'…'` on the call's log), and the text gets the call's reply. The call holds the lane under `live`.
