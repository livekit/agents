# Delegation from a voice agent

A voice agent (the frontend) hands work to a backend agent over A2A and keeps talking while it
runs. The backend's work is slow, reports progress, asks questions in the middle and waits for
approval; the frontend relays all of it, and routes what the user says back to the right task.

| File                | Agent                | What                                                                           |
| ------------------- | -------------------- | ------------------------------------------------------------------------------ |
| `frontend_agent.py` | `delegation-voice`   | The voice agent on the call. Its one tool is the backend.                      |
| `backend_agent.py`  | partition `backend`  | Runs jobs and looks things up: slow, asks questions, confirms, speaks job IDs. |
| `main.py`           |                      | The server: the frontend and the backend added to it.                          |

The backend has three tools: `prepare_job` (slow; asks "normal or high priority?" when the
priority is missing), `submit_job` (reads the job back for approval, then speaks the job ID
exactly) and `look_up` (slow; asks "which quarter?" when the quarter is missing).

## What to expect

Each conversation below is what a user hears, and the behaviour it shows.

**1. A job, start to finish.** Background work, progress, a question, approval, exact words.

> **User:** Run the weekly report export.
> **Agent:** On it, the backend is preparing it.
> *(progress: "checking what it needs", "reserving capacity")*
> **Agent:** Should it run at normal or high priority?
> **User:** High.
> **Agent:** Submit "weekly report export" at high priority?
> **User:** Yes.
> **Agent:** Your job ID is JOB-4821.

**2. Talking while it works.** Small talk stays with the frontend; the open question stays open.

> **User:** Run the weekly report export. ... How long does that usually take?
> **Agent:** Usually under a minute. *(the backend keeps working)*
> **Agent:** Should it run at normal or high priority?

**3. An answer that could go to two places.** Two jobs both ask for a priority; "high" is
ambiguous, so the agent asks which one instead of guessing. Nothing is sent until it is clear.

> **Agent:** Both the export and the cleanup need a priority: normal or high?
> **User:** High.
> **Agent:** For the export, the cleanup, or both?
> **User:** The export.

**4. Two different questions at once.** The export asks for a priority and the lookup for a
quarter. A partial answer settles one and leaves the other open.

> **Agent:** Should the export run at normal or high priority, and which quarter for the audit?
> **User:** High.
> **Agent:** Got it. And which quarter for the audit?

**5. One sentence, two answers.** Each part goes to its own task, once.

> **User:** High for the export, and Q3 for the audit.

**6. Changing your mind.** A correction before the answer was acted on replaces it; after, it
is sent as a correction and the agent says what already happened.

> **User:** High. ... Actually, make it normal.

**7. Saying no.** Nothing is submitted.

> **Agent:** Submit "weekly report export" at high priority?
> **User:** No.
> **Agent:** Okay, I won't submit it.

**8. Messages while the backend is busy (text).** Over the `a2a` text channel the backend uses
the `batch` inbox: messages sent while it is working are merged into one next turn, and every
request gets its answer.

**9. Cancelling one task.** The other keeps going.

> **User:** Cancel the export.
> **Agent:** Cancelled. The audit lookup is still running.

## How it works

- **The backend is a tool.** `RemoteAgent(backend_agent.text).as_tool()`: each call starts a
  remote A2A task and the frontend does not wait for it.
- **Progress.** `ctx.update(...)` in a slow backend tool reaches the frontend as it happens.
- **Questions mid-task.** A tool returns `InputRequired(...)`; the frontend asks it in its own
  words, and the session's routing model (`DelegationOptions(routing_model=...)`) decides which
  open question each thing the user says answers, or that it answers none.
- **Approval.** `submit_job` has `needs_approval=read_back`: the framework reads the job back and
  runs the tool only after the user agreed.
- **Exact words.** `ctx.say(...)` sends the job ID to be spoken as written, across the A2A hop.
- **Inbox per channel.** `inbox={"delegation": "live", "a2a": "batch"}`: requests from the
  frontend each keep their own task and run side by side; text messages are merged.
- **Bound state.** Both entry points bind their store, so tasks, open questions and the
  backend's conversation survive a restart of either side.

See [storage and conversation state](../README.md#storage-and-conversation-state) for the
partitions and bindings.

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
lk agent dev
```

Then join the agent's room from the LiveKit playground, or run `lk agent console` for a
terminal conversation.
