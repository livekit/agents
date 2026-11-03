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
| [models.md](models.md) | Model connection options, timeout phases, errors, retries, and provider streams |
| [telemetry.md](telemetry.md) | Exporting to your backend, application spans, distributed traces, content and PII, LiveKit Cloud recording, and diagnostics |
| [usage.md](usage.md) | Usage snapshots, token and billing units, accounting scope, measurement availability, and performance metrics |
| [context.md](context.md) | Runtime dependencies, persisted fields, explicit load/save, scope, concurrency, and schema evolution |
| [text-session.md](text-session.md) | Session rehydration, backend agent frames, delegation, private histories, and continuation across requests |
| [voice-session.md](voice-session.md) | Audio and turn configuration, agent replacement, frontend expert events, and reply routing |
| [voice-tasks.md](voice-tasks.md) | Constructing and running bounded voice interactions, typed completion, suspension, and cleanup |
| [threads.md](threads.md) | Conversation identity, lifecycle, public history/task access, and the relationship between lanes and session frames |
| [server.md](server.md) | AgentServer/AgentHandler, text and voice entrypoints, handler outcomes, and request policies |
| [clients.md](clients.md) | Thread service and handles, endpoint clients, grants, browser access, and call dispatch |
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

| Document | Contract |
| --- | --- |
| [Storage](internals/storage.md) | Database layout, managed commits, lanes, leases/fences, transport inboxes, and the journal |
| [Rust code map](internals/rust-code-map.md) | Start here for Python-to-Rust execution paths, component entry points and focused test commands |
| [Rust core](internals/rust-core.md) | Crate ownership, deterministic cores, runtime I/O, generated bindings, and verification |
| [Telemetry](internals/telemetry.md) | Data flows, destinations, the span API, user turns, the session report, the naming registry, and how to add telemetry |
| [Telemetry reference](internals/telemetry-reference.md) | Standard conventions, span trees, the complete emitted inventory, consumers, and the Cloud ingest and redaction contract |
| [Python ownership inventory](internals/python-ownership.md) | Complete handwritten framework file coverage, ownership targets, migrated responsibilities, and remaining work |
| [Rust transition inventory](internals/rust-transition-inventory.md) | Python terminology alignment, transition simplification proposals, ordering constraints, and verification |
| [Python file cleanup inventory](internals/python-file-layout-inventory.md) | Handwritten consolidation from 113 to 63 files, familiar public namespaces, utility cleanup, and complete source coverage |
| [Python legacy API inventory](internals/python-legacy-api-inventory.md) | Deprecated arguments, no-op settings, compatibility aliases, concrete replacements, and protocol/data retirement prerequisites |
| [Rust file cleanup inventory](internals/rust-file-layout-inventory.md) | Small-file merge candidates, proposed destinations, and deliberate component boundaries |
| [Lifecycle symmetry audit](internals/rust-lifecycle-symmetry-audit.md) | Enter/exit hooks, event/effect vocabulary, request/receipt pairs, and intentional lifecycle differences |
| [Shared policy audit](internals/rust-shared-policy-audit.md) | Competing core/voice decisions, canonical tool history, policy ownership, and SDK terminology comparison |
| [Rust fuzz crate audit](internals/rust-fuzz-inventory.md) | Simulation structure, contributor navigation, scheduling, contracts, reproduction, and improvement inventory |
| [Type simplification inventory](internals/rust-type-inventory.md) | Redundant audio and Rust aliases, removal candidates, and distinct types to retain |

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
- [`console/`](../console/README.md): a developer tool, not a customer
  frontend. A chat with a front desk, a call over the voice door, and the
  thread's database beside it, read through the admin-only
  [admin API](clients.md#admin-reads).
- [`tests/e2e/`](../tests/e2e/README.md) contains pytest journeys selected with
  `uv run pytest tests/e2e --e2e`: a thread across two workers, interrupt, an
  inbox burst, a worker killed mid-turn, input-required and approvals
  answered on the other worker, a stateless agent serving two threads,
  and the threads service's create and delete.

## Vocabulary

| Term | Meaning |
| --- | --- |
| Agent | Shared behavior configuration; not a conversation or mutable application state |
| Run | One input to a completed result or input-required outcome, including model/tool execution |
| Text session | A logical conversation continued by reconstructing request-scoped execution from explicitly saved state |
| Voice session | A running frontend conversation with audio resources, turn handling, and in-process interaction state |
| Voice task | A temporary voice interaction that returns a typed result and restores its caller |
| Thread | A durable conversation identity (`AT_...`), also used as A2A `contextId` |
| Lane | A configured handler's transport-history/task and concurrency boundary within a thread |
| Frame | One backend agent conversation inside TextSessionState, with its own history and parent-call relationship |
