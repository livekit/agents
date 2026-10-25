# Review: one agent, two doors

A diagnostic example for one agent with two doors and a text during a call ([`docs/delegation.md`](../../docs/delegation.md), Flow D and "When the agent is busy"). One support agent, no back office, with the same four tools on the call and on the text door. Each tool shows one place where the two engines differ, and the call logs every voice commit and which items of the session's history its model never received.

| File               | Agent   | What                                                                                                         |
| ------------------ | ------- | ------------------------------------------------------------------------------------------------------------ |
| `agent.py`         | `doors` | The tools and two `Agent`s on them: `voice` (the session's realtime model) and `chat` (a text model).        |
| `main.py`          |         | The entrypoint: one `AgentHandler` with the call and the `frontend` text door, and the diagnostic logging.   |
| `scenarios/*.yaml` |         | `two_doors`: one lookup, two lookups in one message, a verification, the flaky lookup; text only.           |

The tools:

- `lookup_order(order_id)`: `ctx.update("looking up …")`, 4 s, `ctx.update("almost there")`, 4 s, then the answer. Logs its start, each update and its end with a timestamp.
- `verify_identity()`: `code = await CollectCode()`, then logs `after the await`.
- `verify_identity_safely()`: the same inside `try/except Exception`, which logs what it caught and returns `verification failed`.
- `flaky_lookup()`: raises `ModelRetry("ask for the order number first")`.

`CollectCode` uses the session's realtime model on the call and a text model in text, since an `AgentTask` in a text run has no model unless it names one.

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
uv run python main.py dev
```

Pick "Review: one agent, two doors" in the console; it does both the call and the chat. Every line this example writes starts with `review.frontend`.

```sh
lk agent simulate text --scenarios scenarios/two_doors.yaml --agent-name doors
```

## What the logs say

- `text turn <id>: …`: a text turn's message, each item the run added with its id, and its output.
- `call tool started|updated|ended …` and `call item added: …`: the call.
- `call lane version=N history=K missing from the activity: [ids] missing from the model: [ids]`: after each item on the call, the lane version the call last saw, and the ids of the session's history (`session.history`) that are not in the activity's copy (`_chat_ctx`) and not in the realtime model's context (`_rt_session.chat_ctx`).
- `voice commit: loaded_version=N wrote=[ids] deleted=[ids]`: each voice commit, from the lane as the call loaded it.
- `lookup_order …`, `verify_identity …`, `flaky_lookup …`: the tools.

## What to try, and what to look for

**(a) Progress.** Type `check order A1`.

- `lookup_order … A1: update 'looking up A1' returned at …`: under the `live` inbox the first update lets the text run go on, as on a call. The reply comes at once ("looking it up"), the task stays `working` with `livekit/background` (the console shows it in the background), and the tool keeps running. `end at …` about 8 s later, then a second turn of the same task, with no message, answers it: the console shows the answer arriving on the first task.
- Type something else while the lookup runs: it runs at once, beside the tool.
- By voice, `check order A1`: `call tool updated … 'looking up A1'` and a spoken line at once, then the answer as a later reply after `call tool ended`. Text and call now behave the same.

**(b) Two calls in one step.** Type `check orders A1 and B2`.

- `lookup_order … A1: start at` and `lookup_order … B2: start at` within a few milliseconds: A1's first update let the run go on to B2, and both run in the background. The task waits for both; one later turn answers with both results, about 8 s in all.
- By voice, the same: concurrent, about 8 s.

**(c) A task awaited in a tool.** Type `verify me`, then the code `4821`.

- `CollectCode.record_code … code='4821'` but never `verify_identity … after the await`: in text the `await` raised `TaskAwaited`, the task took over, and the code after it never ran; the task's value became the call's output.
- Type `verify me safely` (on a new chat, or after (c) finished): `verify_identity_safely …: caught TaskAwaited` and the model is told `verification failed`, with no code ever asked. The tool's own `except Exception` swallowed the framework's control flow.
- By voice, `verify me` and `verify me safely`: both log `after the await: code=…`; on a call the tool resumes with the value.

**(d) `ModelRetry`.** Type `run the flaky lookup`.

- `text turn …: added out:flaky_lookup='ask for the order number first'`, then an assistant message that asks for the order number: in text the reason reached the model.
- By voice: `call tool ended …: status=error message='An internal error occurred'` (or the same text as the tool's output): on a call the reason is lost.

**(e) A text during a call.** Start a call and say `hi`. In the console chat, type `my order is Z9`. Then ask by voice `what did I text you?`.

- With the text door under `live` (the default) the call holds the lane for the whole call and answers the text itself: `call item added: user:'my order is Z9'` on the call's log, then the call's reply, which is also the text's answer in the chat. No `text turn` line: the text door's handler did not run.
- `missing from the model: []` on the following `call lane version=` lines: the text reached the realtime model's context.
- The voice answer to `what did I text you?` knows `Z9`.
- `voice commit: … deleted=[]`: the call deletes only what it removed itself, so a text's items are never reverted.
- With `inbox="queue"` on the text door in `main.py` the call takes the lane per voice turn: the text door answers the text, and the call reloads its history on the next voice turn.
