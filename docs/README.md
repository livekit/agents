# LiveKit Agents 2.0: the backend API

An agent is configuration, built once and shared. A thread is one durable
conversation with a person, across channels and over time. A run is one
input to one result. Every document below is one feature, from the API's
point of view.

## How to read this folder

Each feature document describes its API and behavior. Companion pages expand
on related capabilities. The examples show applications of these features.

## Principles

**Typed end to end, in Python and in Node.**

- No `Any` in the public API: every function, class and callback is fully
  typed, user code passes mypy strict and pyright, and the Node package
  ships `.d.ts` files with no `any`.
- Generics carry the types: `Agent[Context]`,
  `RunContext[Context]` and typed run results. A typo in `ctx.context.x` is a
  type error.
- Data at every boundary is a validated type: persisted application data, output and tool arguments are validated schemas
  in Python and typed in Node. Local context can be an arbitrary Python object
  and is not automatically serialized.
- The wire is typed: A2A frames, grants, threads service calls and extension
  metadata are records generated from one declaration and shared by Python
  and Node.
- CI enforces it: the examples type-check (mypy strict, `tsc`), type tests
  pin the generics, and generated stubs must match their Rust declarations.

## Documents

| Document | Feature |
| --- | --- |
| [agents.md](agents.md) | `Agent`: definition, runs and results, context, agents as tools, local and remote delegation |
| [planned-agents.md](planned-agents.md) | `iter`, filtered/prefixed toolsets, history processors, model-call hooks, capabilities |
| [tools.md](tools.md) | `@agent.tool`, `@function_tool`, toolsets, MCP, approvals, errors |
| [state-and-deps.md](state-and-deps.md) | typed context; explicit field load/save, scopes, and fresh runtime dependencies |
| [human-in-the-loop.md](human-in-the-loop.md) | `InputRequired`, approvals, and the other transitions |
| [agent-router.md](agent-router.md) | agents as tools, typed completion, human input, frames and histories |
| [threads.md](threads.md) | what a thread is, how it is found, what it holds, `ctx.thread`, lifecycle |
| [lanes.md](lanes.md) | one agent's part of a thread: history, state, the lease, run order |
| [storage.md](storage.md) | what is saved, the backends, the unit of work |
| [journal.md](journal.md) | what the framework records about a thread, separate from the history |
| [text-endpoints.md](text-endpoints.md) | `@server.a2a_session`, turn policies, bursts and interruptions |
| [channels.md](channels.md) | where a message came from and what that changes |
| [planned-channels.md](planned-channels.md) | SMS, WhatsApp, email, outbound text |
| [a2a.md](a2a.md) | the wire reference: the agent card, tasks, metadata keys, errors |
| [a2a-live-ext.md](a2a-live-ext.md) | the LiveKit extension's specification |
| [voice.md](voice.md) | `LiveSession`, the voice door, one thread for calls and texts |
| [backend-api.md](backend-api.md) | the threads service, its history and tasks, thread grants, `AgentClient` |
| [planned-backend-api.md](planned-backend-api.md) | `lk.threads`, watching a thread, the Node client |
| [rust-core.md](rust-core.md) | the Rust core: crates, boundaries, bindings, testing, codebase rules |
| [planned-rust-core.md](planned-rust-core.md) | run ownership, host bindings, deterministic cores, runtime I/O, and verification |
| [examples/use-cases/](examples/use-cases/README.md) | agent use cases, one idea each |

## Running it

- [`examples/`](../examples/README.md): three fast brain and slow brain
  pairs, each one worker: a dental practice's front desk, a hotel
  receptionist, an airline's customer care, with simulation scenarios.
- [`console/`](../console/README.md): the frontend. A chat with a front
  desk, a call over the voice door, and the thread's database beside it.
- `tests/test_v0_checks.py` runs the end-to-end checks against that stack
  (`V0_STACK=1`): a thread across two workers, resubscribe, interrupt, an
  inbox burst, a worker killed mid-turn, input-required and approvals
  answered on the other worker, a stateless agent serving two threads,
  the threads service's create and delete, and the console's spoken call.

## Words

- **thread**: the conversation. It has an id the platform mints, a
  history, explicitly saved application fields, and A2A tasks. The A2A `contextId` is its id.
- **lane**: one agent's part of a thread. Each agent in a conversation has
  its own transport history and A2A tasks there, written by one run at a time; local router frames remain separate.
- **run**: one input to one result, tools included. A text message, a voice
  turn. The framework loads and commits transport records; application fields and routing state require explicit load/save.
- **transition**: what a tool or a hook returns to say what happens next:
  `Completed`, `InputRequired`, `Working`, `Failed`, `Rejected`.
