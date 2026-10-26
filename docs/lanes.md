# Lanes

A lane is an endpoint or handler's storage and concurrency boundary within a
thread. `AgentHandler(lane=...)` selects it; the worker's configured agent
name is the default. It holds transport history and tasks. A local agent
invocation does not allocate a lane merely because it has another agent ID.

Local routing frames and their private histories are saved through
application-owned router state. Several frames may reference the same agent
definition while remaining separate conversations.

```
thread AT_...                one database
├─ lane "front-desk"         history, application data, tasks of the front desk
├─ lane "eligibility"        the same for the expert it delegates to
└─ journal                   every event from every lane, append-only
```

## What a lane holds

|             |                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------- |
| history     | every message, tool call and result the agent's model sees, item by item                          |
| application data | explicitly saved fields and router state, with context-scope identity; see [state-and-deps.md](state-and-deps.md) |
| tasks       | one row per transport task, as A2A tasks                                            |
| inbox       | messages received while a turn ran, under the `batch` policy                                      |
| the lease   | which worker holds the lane right now, and until when                                             |

The journal and the database are per thread and shared by every lane; the
rest is per lane. See [storage.md](storage.md) for the tables and
[journal.md](journal.md) for the journal.

A phone line and text endpoint configured for the same lane share transport
history. Their application data remains subject to explicit load/save.

Each direct `agent.run()` executes the selected agent. An optional
[AgentRouter](agent-router.md) owns local routing frames and private histories;
the application stores that routing state explicitly. Remote endpoints using
the same project thread own their configured lanes.

Context sharing is explicit. Shared persisted scopes serialize updates;
separate child contexts have independent fields. See
[application context](state-and-deps.md).

## One writer at a time

A lane has exactly one writer at a time, held two ways:

- **The lease.** A run takes the lane's lease before it loads anything and
  keeps it until its commit. The lease lives for 10 seconds and the holder
  renews it every 3; both are options of `AgentDBStorage`. A worker that
  dies stops renewing, and the lease lapses within its lifetime. A holder
  whose renewal arrives late keeps the lane if nobody took it meanwhile:
  the lease token, not the clock, is what another run has to replace.
- **The fence.** Every commit is conditional on the version the run loaded
  and on its lease token, and bumps the version. A run that outlived its
  lease finds the lane moved on, gets `VersionConflict`, and writes nothing.

The lease keeps two runs from executing at once; the fence keeps a run that
lost its lease from writing over the one that took it.

Runs waiting for a lane execute in the order their messages arrived: each
message's `submitted` row is written before its run waits for the lease,
and the rows' order is the order the waiters run in. Under `interrupt` and
`queue`, a waiter that has not taken the lane within 60 seconds gives up,
and its task fails with `livekit/reason` `lease-timeout`. Under `batch` a
waiter keeps waiting: the run that drains the inbox owns its messages.

A run whose worker died is not run again. Its task fails `worker-lost` as
soon as the next holder finds it, and the client resends the message.

The `concurrent` policy is the one exception: its runs take no lease and
commit append-only, with no fence, and write no state.

## Voice and text share a lane

A call's `LiveSession`, given `thread=ctx.thread`, keeps the conversation
in memory and takes the lane's lease for each voice turn, from the end of the person's turn to the
commit of the reply, then releases it. A text that arrives between two voice
turns runs on any worker that takes the lane. When the call's next turn
takes the lease and finds a newer version than the one it last committed,
it reloads the transport history. Application code must reload marked context
fields before using updates saved by another activation.

## Two brains, one thread

When a call delegates to an expert that shares the caller's thread (its
card offers the Live extension; see [multi-agent composition](agents.md#working-with-other-agents)), the
caller's worker writes the caller's lane and the expert's worker writes the
expert's. Neither writes the other's rows. The expert's result reaches the caller through the delegate tool; the
expert's own turns stay in its lane. A tool reads another agent's history
with `ctx.thread.lane(agent).history()`; the journal, shared, orders every
lane's events in one sequence.
