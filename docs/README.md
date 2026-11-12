# LiveKit Agents 2.0

An `Agent` defines behavior. `TextSession` coordinates a conversation across
requests; `VoiceSession` runs an active voice conversation. Application context
carries data and services, with explicit persistence for selected fields.
These documents define the intended API and behavior.

## Start here

1. [Agents](agents.md): define behavior, register a tool, run an invocation, and read its result.
2. [Tools](tools.md): tool execution, human input, approvals, and agents as tools.
3. [Context](context.md): construct dependencies and explicitly load/save application data.
4. Choose [TextSession](text-session.md) for request-scoped text conversations or [VoiceSession](voice-session.md) for a live call.

For bounded voice interactions, continue with [VoiceTask](voice-tasks.md).
To expose and call your application, read [server entrypoints](server.md) and
[clients](clients.md).

## Public API

| Document | Owns |
| --- | --- |
| [agents.md](agents.md) | Agent configuration, RunContext, execution, output contracts, history, lifecycle hooks, and streaming |
| [tools.md](tools.md) | Tools, toolsets/MCP, approvals, human input, errors, and local/remote agents as tools |
| [models.md](models.md) | Model strings and the inference catalog, realtime options, connection options and pools, the HTTP/WebSocket client, timeouts, errors, retries, provider streams, and the plugin-author contract |
| [telemetry.md](telemetry.md) | Exporting to your backend, application spans, distributed traces, content and PII, LiveKit Cloud recording, Prometheus, and diagnostics |
| [usage.md](usage.md) | Usage snapshots, token and billing units, accounting scope, measurement availability, and performance metrics |
| [context.md](context.md) | Runtime dependencies, persisted fields, explicit load/save, scope, concurrency, and schema evolution |
| [text-session.md](text-session.md) | Session rehydration, backend agent frames, delegation, private histories, and continuation across requests |
| [voice-session.md](voice-session.md) | Audio and turn configuration, agent replacement, frontend expert events, and reply routing |
| [lexer.md](lexer.md) | A model reply's text on its way to the voice, the transcript and the history: hidden reasoning and template leaks, expressive markup, markdown and emoji, transcript sync, and how long text is held |
| [voice-tasks.md](voice-tasks.md) | Constructing and running bounded voice interactions, typed completion, suspension, and cleanup |
| [threads.md](threads.md) | Conversation identity, lifecycle, public history/task access, and the relationship between partitions and session frames |
| [server.md](server.md) | AgentServer/AgentHandler, text and voice entrypoints, handler outcomes, and request policies |
| [clients.md](clients.md) | Thread service and handles, endpoint clients, grants, browser access, and call dispatch |
| [auth.md](auth.md) | Principals, grants, how a thread id travels with its token, agent-db access, and endpoint verification |
| [channels.md](channels.md) | Platform ingress/egress, SMS/WhatsApp/email, attachments, and cross-channel continuation |
| [testing.md](testing.md) | Run event assertions, conversation judges, simulations, and SimulationContext verdicts and callbacks |

Each contract has one owning page. Other pages show how to use it and link to
its definition. Run results and streaming live in `agents.md`; wire encoding
and storage mechanics have separate references below.

## Protocol

| Document | Contract |
| --- | --- |
| [A2A](protocol/a2a.md) | Agent cards, wire tasks, metadata, tokens, errors, and third-party interoperability |

## Internals

The [contributor index](internals/README.md) organizes the implementation guides by topic.
Start with the [execution paths](internals/architecture.md#execution-paths) to follow an execution path, or
[development](internals/development.md) to build and check the framework.

## Principles

**Typed end to end, in Python and in Node.**

- No `Any` in the public API: every function, class and callback is fully
  typed, user code passes mypy strict and pyright, and the Node package
  ships `.d.ts` files with no `any`.
- Generics carry the types: `Agent[Context, Output]`, inferred from
  `context_type=` and `output_type=`, `RunContext[Context]` and typed run
  results. A typo in `ctx.context.x` is a type error.
- Data at every boundary is a validated type: persisted application data, output and tool arguments are validated schemas
  in Python and typed in Node. Local context can be an arbitrary Python object
  and is not automatically serialized.
- The wire is typed: A2A frames, grants, threads service calls and LiveKit
  metadata are records generated from one declaration and shared by Python
  and Node.
- CI enforces it: the examples type-check (mypy strict, `tsc`), type tests
  pin the generics, and generated stubs must match their Rust declarations.

## Running the examples

- [`examples/`](../examples/README.md): three fast brain and slow brain
  pairs, each one worker: a dental practice's front desk, a hotel
  receptionist, an airline's customer care, with simulation scenarios.
- [`python/tests/e2e/`](../python/tests/e2e/README.md) contains pytest journeys selected with
  `uv run pytest python/tests/e2e --e2e`: a thread across two workers, interrupt, an
  inbox burst, a worker killed mid-turn, input-required and approvals
  answered on the other worker, a stateless agent serving two threads,
  and the threads service's create and delete.

## Vocabulary

| Term | Meaning |
| --- | --- |
| Agent | Shared behavior configuration; not a conversation or mutable application state |
| Run | One input to a completed result or input-required outcome, including model/tool execution |
| Text session | Request-scoped conversation execution continued through an explicit snapshot or store binding |
| Voice session | A running frontend conversation with audio resources, turn handling, and in-process interaction state |
| Voice task | A temporary voice interaction that returns a typed result and restores its caller |
| Thread | A durable conversation identity (`AT_...`), also used as A2A `contextId` |
| Partition | History, routing, inbox and write ownership selected by a partition_key within a thread |
| Frame | One backend conversation inside a text SessionState, with its own history and parent-call relationship |
