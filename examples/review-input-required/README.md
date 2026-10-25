# Review: a question to the person

A diagnostic example for questions to the person ([`docs/delegation.md`](../../docs/delegation.md), Flows C and D, "When the expert asks a question") and what the expert sees. A dental front desk delegates to a back office whose tool asks for an email with `InputRequired`, and whose newsletter flow is an `AgentTask`. Both sides log what the framework did with the question and the answer.

| File                | Agent           | What                                                                                                    |
| ------------------- | --------------- | ------------------------------------------------------------------------------------------------------- |
| `frontend_agent.py` | `inputreq`      | The front desk, on the phone and by text (the `frontend` endpoint). Its one tool is a `LoggingDelegate`. |
| `backend_agent.py`  | lane `backend`  | The back office: `book_visit` asks for the email, `newsletter_signup` awaits the `CollectEmail` task.    |
| `review_log.py`     |                 | How items (`user:'…'`, `call:ask({…})`, `out:…`) and task rows (`id:status+ask(…)`) are printed.         |
| `main.py`           |                 | The one entrypoint: the server, the two handlers added to it.                                           |
| `scenarios/*.yaml`  |                 | `input_required`: the email given, a question given instead, the newsletter task.                        |

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
uv run python main.py dev
```

Pick "Review: a question to the person" in the console. Every line this example writes starts with `review.frontend` or `review.backend`; `grep -E 'review\.(frontend|backend)'` on the worker's output shows only them.

```sh
lk agent simulate text --scenarios scenarios/input_required.yaml --agent-name inputreq
```

## What the logs say

- `delegate ask <call_id>: request=…` and `delegate ask <call_id>: input_required=…` or `answer=…`: every call of the front desk's `ask` tool, on both doors.
- `frontend turn <id>: added …`: each item a text run added, in order. The person's answer to a question shows as a `call:ask({"request": …})` with no assistant message before it: the framework made that call, with the person's words, to continue the back office's task.
- `call tool started|updated|ended …` and `call item added: …`: the same on the call.
- `backend turn <id>: …`: at the start of every back-office turn, its task id, the channel and message, the own history (size and last two items), whether `caller_history` came, the caller's lane it was told (`caller_lane=`), that lane as stored (whole), its own task rows and the front desk's, newest first. A row prints as `a1b2c3:input-required+ask(call_xyz)`, or `+ask(no call_id)` for an ask with no open call, plus `[queued=…]`. At the end, `output=` and `input_required=`.
- `backend book_visit <call_id>: day=… email=…`: every call of the tool, with its arguments.

## What to try, and what to look for

**(a) A question answered with a question.** By text: `Book me a cleaning on Monday`. The back office asks for the email. Then type `why do you need my email?`.

- The console shows the second message going to the task that asked: the second response has the same task id as the first, and the front desk's queue panel shows the question in front of the person until it is answered.
- Frontend, second turn: `added call:ask({"request": "why do you need my email?"})` is the first item, before any `assistant:` item: the input went, as typed, to the back office, with no front desk model call. `delegate ask call_…` logs the request.
- Backend, second turn: `task=` is the same id as the first back-office turn, and `own rows:` shows that row `working` (picked up, no new row). The own history's last item is `out:book_visit='why do you need my email?'`: the input was written as the tool's output. The back office model answers the question and asks again: its task goes back to `input-required`, and the follow-up is in front of the person at once.
- Then type the email, `ana@example.com`: the same path; `backend book_visit …: email='ana@example.com'`, `output='…booked…'`, and the back office row ends `completed` with no `mergedInto`.

**(b) The same by voice.** Start a call, say `book me a cleaning on Monday`, then answer the email question with `why do you need my email?`.

- `call tool started call_…: ask({"request": "why do you need my email?"})` right after `call item added: user:'…'`: the front desk made the call with your transcript, without asking its model (no forced `tool_choice`). The realtime model may also say a few words on that turn; the front desk's instructions keep them short. That is a known limit.

**(c) An ask with no pending call.** By text: `sign me up for the newsletter`, then give `cara@example.com`.

- `backend newsletter_signup call_…: awaiting CollectEmail`; in text the tool stops at the `await` (`TaskAwaited`) and the task asks.
- `own rows:` shows the turn as `…:input-required+ask(no call_id)`. The answer continues that task by its id: the row ends `completed` after `CollectEmail.record_email`, instead of staying open forever.

**(d) Ask by text, answer by voice, then text again.** By text: `Book me a cleaning on Monday` (the question comes back). Start a call and say `my email is ana@example.com`. Hang up and type `thanks`.

- On the call, `call tool started call_…: ask({"request": "my email is ana@example.com"})`: the call started with the lane's open question in front of the person, and your answer went to the back office.
- After the call, the front desk row of the first text turn is `completed`, and `thanks` is a new task: no question is in front any more.
- The reverse: ask by voice (`book me a cleaning on Monday`), hang up before answering, then text the email. The call left its question on the lane as an `input-required` row, and the text continues it.

**(e) Two questions at once.** On a call: `book me a cleaning on Monday and sign me up for the newsletter`. The front desk calls `ask` twice; both questions come back.

- Only one is spoken; the other waits (`[queued=True]` on its row if the call ends first). Answer the first; the second is asked after the answer to the first is spoken.

**(f) What the expert sees.** On every backend turn:

- `caller_history=none` and `caller_lane=inputreq`: with a shared thread the delegate sends no conversation, and names its lane.
- `inputreq lane as stored` ends with the person's message that led to this delegation: the front desk wrote it before it sent the delegation, so the back office can read the person's own words.
