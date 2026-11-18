# Delegation from a voice agent

The phone line of Bistro Lumen, a made-up restaurant, as a fast brain and two slow brains: a voice host that talks to the
guest, a reservations desk and a pastry kitchen that do slow work in the background, ask
questions mid-task and report progress while the call goes on.

| File              | Agent                     | What                                                                                     |
| ----------------- | ------------------------- | ---------------------------------------------------------------------------------------- |
| `main.py`         | `delegation-voice`        | The voice host on the call. Its two tools are the remote desks. The server, all handlers. |
| `reservations.py` | partition `reservations`  | Finds and holds tables: slow, asks for seating, reads the hold back, speaks the code.    |
| `pastry.py`       | partition `pastry`        | Plans a cake: slow, independent work that runs beside a table search.                    |

## A call

> **Guest:** A table for four tonight at eight, please. It's my wife's birthday.
> **Host:** Lovely, I'm checking the book now. Would you like a cake as well?
> *(the reservations desk is searching; its progress reaches the host as it happens)*
> **Guest:** Yes, chocolate if possible. Do you have vegetarian dishes?
> **Host:** The menu is seasonal French with vegetarian mains. The pastry kitchen is on the cake.
> **Host:** We have room inside and on the terrace. Which would you like?
> **Guest:** The terrace.
> **Host:** Shall I hold a terrace table for four tonight at eight, with a 20 euro deposit?
> **Guest:** Yes.
> **Host:** Your confirmation code is LUMEN-482. The chocolate cake will be ready for dessert.

## What it shows

- **Remote work off the floor.** Each desk is a `RemoteAgent(...).as_tool()`: a call starts a
  remote A2A task and the host keeps talking. The tables search and the cake run at the same
  time, as two independent tasks.
- **Progress while it works.** `ctx.update(...)` in a slow tool streams to the host as it
  happens; the host decides whether and how to mention it, and a result makes stale progress
  moot.
- **A question in the middle of the work.** `find_table` returns `InputRequired` when the
  seating is missing. The host asks in its own words; the session's routing model
  (`DelegationOptions(routing_model=...)`) recognizes "the terrace" as the answer to that exact
  task and sends it there, while a side question about the menu stays with the host.
- **Approval held by the framework.** `hold_table` has `needs_approval=read_back`: the desk reads
  the hold back word for word and the deposit is taken only after the guest agreed.
- **Exact words across a hop.** `ctx.say(...)` sends the confirmation code to be spoken as
  written; the host does not rephrase it.
- **Merged follow-ups.** The reservations desk uses `inbox="batch"`: when the guest changes
  their mind while a turn runs ("make it six"), the queued messages are merged into the next
  turn, and the earlier tasks complete pointing at it (`livekit/mergedInto`).
- **Bound state.** Every door binds its store, so the call's remote tasks, open questions and
  the desks' conversations survive a restart of either side.

See [storage and conversation state](../README.md#storage-and-conversation-state) for the
partitions and bindings.

## Run

With the stack and the environment of [`../README.md`](../README.md), from this directory:

```sh
lk agent dev
```

Then join the agent's room from the LiveKit playground, or run `lk agent console` for a
terminal conversation.
