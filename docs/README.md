# LiveKit Agents 2.0: the backend API

Status: in development on the `v2.0` branch. Nothing here is released.

An agent is configuration, built once and shared. A thread is one durable
conversation with a person, across channels and over time. A run is one
input to one result. Every document below is one feature, from the API's
point of view.

## How to read this folder

- A document with no prefix describes the target API; the table below marks
  built features and end-state designs whose implementation is still in progress.
- A document named `planned-<topic>.md` describes design that is not built
  yet. It extends the built document of the same topic.
- [TODO.md](TODO.md) lists the places where the code still differs from a
  built document. It shrinks to nothing as the code lands.
- The same rule holds for the use cases: `planned-NN_*.py` uses APIs that
  are not built.

## Principles

**Typed end to end, in Python and in Node.**

- No `Any` in the public API: every function, class and callback is fully
  typed, user code passes mypy strict and pyright, and the Node package
  ships `.d.ts` files with no `any`.
- Generics carry the types: `Agent[Context]`,
  `RunContext[Context]` and typed run results. A typo in `ctx.context.x` is a
  type error. Context/output inference signatures are implementation work.
- Data at every boundary is a validated type: persisted application data, output and tool arguments are validated schemas
  in Python and typed in Node. Local context can be an arbitrary Python object
  and is not automatically serialized.
- The wire is typed: A2A frames, grants, threads service calls and extension
  metadata are records generated from one declaration and shared by Python
  and Node.
- CI enforces it: the examples type-check (mypy strict, `tsc`), type tests
  pin the generics, and generated stubs must match their Rust declarations.

## Documents

| Document | Status | Feature |
| --- | --- | --- |
| [agents.md](agents.md) | target plus implementation notes | `Agent`: stateless definition, `run()`, `RunContext`, structured output, validators |
| [planned-agents.md](planned-agents.md) | planned | `run_stream`, `iter`, `toolsets=`, history processors, model-call hooks, capabilities |
| [tools.md](tools.md) | built | `@agent.tool`, `@function_tool`, toolsets, MCP, approvals, errors |
| [state-and-deps.md](state-and-deps.md) | end-state design | typed context; automatic persisted fields with field-name storage, scopes, and fresh runtime dependencies |
| [human-in-the-loop.md](human-in-the-loop.md) | built | `InputRequired`, approvals, and the other transitions |
| [agent-tasks.md](agent-tasks.md) | end-state design | conversational tasks: draft state, automatic input waits, typed final output |
| [code-workflows.md](code-workflows.md) | deferred | future graphs/workflows: requirements and extension points, no committed authoring API |
| [threads.md](threads.md) | built | what a thread is, how it is found, what it holds, `ctx.thread`, lifecycle |
| [lanes.md](lanes.md) | built | one agent's part of a thread: history, state, the lease, run order |
| [storage.md](storage.md) | built | what is saved, the backends, the unit of work |
| [journal.md](journal.md) | built | what the framework records about a thread, separate from the history |
| [text-endpoints.md](text-endpoints.md) | built | `@server.a2a_session`, turn policies, bursts and interruptions |
| [channels.md](channels.md) | built | where a message came from and what that changes |
| [planned-channels.md](planned-channels.md) | planned | SMS, WhatsApp, email, outbound text |
| [a2a.md](a2a.md) | built | the wire reference: the agent card, tasks, metadata keys, errors |
| [a2a-live-ext.md](a2a-live-ext.md) | built | the LiveKit extension's specification |
| [voice.md](voice.md) | built | `LiveSession`, the voice door, one thread for calls and texts |
| [multi-agent.md](multi-agent.md) | built | delegation, handoff, local and remote experts |
| [backend-api.md](backend-api.md) | built | the threads service, its history and tasks, thread grants, `AgentClient` |
| [planned-backend-api.md](planned-backend-api.md) | planned | `lk.threads`, watching a thread, the Node client |
| [rust-core.md](rust-core.md) | built | the Rust core: crates, boundaries, bindings, testing, codebase rules |
| [planned-rust-core.md](planned-rust-core.md) | planned | the run unit of work in Rust, binding moves, CI checks |
| [examples/use-cases/](examples/use-cases/README.md) | mixed | fourteen complete agents, one idea each |

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
  history, a typed state, and tasks. The A2A `contextId` is its id.
- **lane**: one agent's part of a thread. Each agent in a conversation has
  its own history, state and tasks there, written by one run at a time.
- **run**: one input to one result, tools included. A text message, a voice
  turn. The framework loads the lane before a run and saves it after.
- **transition**: what a tool or a hook returns to say what happens next:
  `Completed`, `InputRequired`, `Working`, `Failed`, `Rejected`.
