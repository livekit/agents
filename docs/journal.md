# The journal

Every thread keeps two records of itself. The **history** is what the model
sees: messages, tool calls and results, one per lane, owned by the developer
and editable, since truncating it, inserting a summary or deleting an item
are the developer's tools for managing context. The **journal** is what
happened, owned by the framework and append-only: who ran, when, on which
channel, what a tool did. A developer cannot edit it, which is the point.

## Why it exists

- **Derivations that must be trusted.** A run that started and never ended,
  the channel each request came on, the order of events across agents: all
  derived from the journal, never from the editable history. Who is
  speaking is the lane's agent stack, written by the same commit.
- **A record that survives compaction.** After a long conversation is
  summarized, the history is short and the journal still holds every turn.
- **One stream for everything that watches.** The journal's events are the
  same `AgentSessionEvent` objects the voice runtime emits to the console,
  dashboards and evals. A thread opened later replays the same objects
  production recorded.
- **Debugging a thread after the fact.** Opening a thread's journal shows
  what the agent did and why, tool by tool, including what a crash left
  unfinished.

## What is in it

One journal per thread, shared by every lane. Each event carries:

| Field     | What                                              |
| --------- | ------------------------------------------------- |
| `id`      | its position in the thread's journal             |
| `ts_ms`   | when it was written, in milliseconds              |
| `lane`    | the agent it belongs to                           |
| `run_id`  | the run it happened in                            |
| `type`    | one of the types below                            |
| `payload` | the event, as JSON                                |

| Event                     | What it records                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `run_started`             | a run began: its task id, its answer stream, the channel. A `run_started` with no `run_ended` after it is how an interrupted run is found |
| `run_ended`               | how the run ended: `outcome` (`completed`, `input-required`, `failed`, `canceled`) and `reason`                                        |
| `usage`                   | the run's model requests, tokens and tool calls                                                                                        |
| `conversation_item_added` | an item entered the history: a message, a function call, a function call output, a handoff                                            |
| `tool_execution_updated`  | one call's lifecycle: started, a progress update from `ctx.update()`, ended                                                           |
| `error`                   | a failure the runtime recorded                                                                                                        |

On a call, the voice runtime's own events are recorded as well: agent and
user state changes, transcripts, interruptions, end-of-turn and
answering-machine detection. They are the same `AgentSessionEvent`
payloads the console shows and are not listed here.

## Reading it

Inside a run, `await ctx.thread.journal(after_id=0, limit=100)` returns one
page in order; pass the last `id` to read the next. The journal is not
readable from outside the agent; it is an operations view, and the console
reads it.

## What is not in it

Audio and images; the state object, which is a snapshot on the lane rather
than a stream of edits; and anything the developer writes to the history by
hand, which is why the two are separate.
