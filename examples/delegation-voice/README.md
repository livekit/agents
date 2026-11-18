# Delegation from a voice agent

A voice assistant (the fast brain) with two remote agents (slow brains) as its tools. The
backends do slow work in the background, report progress, ask questions mid-task and wait for
approval, while the call goes on.

| File        | Agent                | What                                                                          |
| ----------- | -------------------- | ----------------------------------------------------------------------------- |
| `main.py`   | `delegation-voice`   | The voice assistant on the call; its tools are the two backends. The server.  |
| `jobs.py`   | partition `jobs`     | Prepares and submits jobs: slow, asks for the priority, confirms, speaks the ID. |
| `lookup.py` | partition `lookup`   | Looks a topic up: slow, independent work that runs beside a job.             |

## A call

> **User:** Run the weekly report export.
> **Assistant:** On it, the jobs backend is preparing it.
> *(the jobs backend is working; its progress reaches the assistant as it happens)*
> **User:** Meanwhile, what do we have on the Q3 audit?
> **Assistant:** I'll look that up too.
> **Assistant:** Should the export run at normal or high priority?
> **User:** High.
> **Assistant:** Submit 'weekly report export' at high priority?
> **User:** Yes.
> **Assistant:** Your job ID is JOB-4821. There are three documents on the Q3 audit; the most recent is from last week.

## What it shows

- **Work off the floor.** Each backend is a `RemoteAgent(...).as_tool()`: a call starts a
  remote A2A task and the assistant keeps talking. A job and a lookup run at the same time,
  as two independent tasks.
- **Progress.** `ctx.update(...)` in a slow tool streams to the assistant as it happens; the
  assistant decides whether and how to mention it.
- **A question mid-task.** `prepare_job` returns `InputRequired` when the priority is missing.
  The assistant asks in its own words; the session's routing model
  (`DelegationOptions(routing_model=...)`) recognizes "high" as the answer to that task and
  sends it there, while unrelated requests stay with the assistant.
- **Approval.** `submit_job` has `needs_approval=read_back`: the backend reads the job back word
  for word and submits only after the user agreed.
- **Exact words across a hop.** `ctx.say(...)` sends the job ID to be spoken as written.
- **Merged follow-ups.** The jobs backend uses `inbox="batch"`: messages that arrive while a
  turn runs are merged into the next turn, and the earlier tasks complete pointing at it
  (`livekit/mergedInto`).
- **Bound state.** Every entry point binds its store, so remote tasks, open questions and the
  backends' conversations survive a restart of either side.

See [storage and conversation state](../README.md#storage-and-conversation-state) for the
partitions and bindings.

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
lk agent dev
```

Then join the agent's room from the LiveKit playground, or run `lk agent console` for a
terminal conversation.
