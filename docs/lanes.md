# Lanes

A lane is one agent's slice of a thread. Every registered agent that takes
part in a thread gets one, created by its first commit, and everything that
agent knows about the conversation lives in it.

```
thread AT_...                one database
├─ lane "front-desk"         history, state, agent stack, tasks of the front desk
├─ lane "eligibility"        the same for the expert it delegates to
└─ journal                   every event from every lane, append-only
```

## What a lane holds

|             |                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------- |
| history     | every message, tool call and result the agent's model sees, item by item                          |
| state       | the agent's typed state; see [state-and-deps.md](state-and-deps.md)                               |
| agent stack | who is speaking: the agent's name, or an `AgentTask` class and its arguments, pushed on a handoff |
| tasks       | one row per turn and per background task, as A2A tasks                                            |
| inbox       | messages received while a turn ran, under the `batch` policy                                      |
| the lease   | which worker holds the lane right now, and until when                                             |

The journal and the database are per thread and shared by every lane; the
rest is per lane. See [storage.md](storage.md) for the tables and
[journal.md](journal.md) for the journal.

A lane is per agent, not per door. An agent's phone line and its text
endpoint write the same lane, which is how a call knows what was said by
text and a text knows what was said on the call. Handoffs between agents on
one worker stay inside one lane and move its agent stack. The framework
routes: the next message, by voice or by text, runs the agent on top of the
stack; in text, `root.run(message)` on the root a handler names runs
whichever agent is on top. A lane boundary
is a delegation to another registered agent: two brains in one thread are
two lanes, whether the second brain is a worker of its own or an
`AgentHandler(lane="backend")` of the same worker.

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
it reloads the lane's history and state before the model reads them, so the
reply knows about the text.

## Two brains, one thread

When a call delegates to an expert that shares the caller's thread (its
card offers the Live extension; see [multi-agent.md](multi-agent.md)), the
caller's worker writes the caller's lane and the expert's worker writes the
expert's. Neither writes the other's rows. The expert's result reaches the caller through the delegate tool; the
expert's own turns stay in its lane. A tool reads another agent's history
with `ctx.thread.lane(agent).history()`; the journal, shared, orders every
lane's events in one sequence.
